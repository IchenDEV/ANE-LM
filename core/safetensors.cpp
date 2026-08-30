#include "safetensors.h"
#include <ane_lm/common.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits>
#include <nlohmann/json.hpp>

namespace ane_lm {

static SFDtype parse_dtype(const char* s, int len) {
    if (len == 4 && memcmp(s, "BF16", 4) == 0) return SFDtype::BF16;
    if (len == 3 && memcmp(s, "F16", 3) == 0) return SFDtype::F16;
    if (len == 3 && memcmp(s, "F32", 3) == 0) return SFDtype::F32;
    if (len == 3 && memcmp(s, "F64", 3) == 0) return SFDtype::F64;
    if (len == 3 && memcmp(s, "I32", 3) == 0) return SFDtype::I32;
    if (len == 3 && memcmp(s, "I64", 3) == 0) return SFDtype::I64;
    if (len == 2 && memcmp(s, "U8", 2) == 0) return SFDtype::U8;
    return SFDtype::Unknown;
}

int SafeTensors::dtype_size(SFDtype d) {
    switch (d) {
        case SFDtype::BF16: case SFDtype::F16: return 2;
        case SFDtype::F32: case SFDtype::I32: return 4;
        case SFDtype::F64: case SFDtype::I64: return 8;
        case SFDtype::U8: return 1;
        default: return 0;
    }
}

bool SafeTensors::parse_header(const char* bytes, int64_t json_len) {
    n_tensors_ = 0;
    auto header = nlohmann::json::parse(bytes, bytes + json_len, nullptr, false);
    if (header.is_discarded() || !header.is_object()) return false;

    const size_t data_bytes = mmap_size_ - 8 - header_size_;
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        if (n_tensors_ >= SF_MAX_TENSORS || it.key().size() >= SF_MAX_NAME || !it.value().is_object()) {
            return false;
        }

        const auto& descriptor = it.value();
        if (!descriptor.contains("dtype") || !descriptor["dtype"].is_string()
            || !descriptor.contains("shape") || !descriptor["shape"].is_array()
            || !descriptor.contains("data_offsets") || !descriptor["data_offsets"].is_array()
            || descriptor["data_offsets"].size() != 2) {
            return false;
        }

        SFTensor* tensor = &tensors_[n_tensors_];
        memset(tensor, 0, sizeof(*tensor));
        memcpy(tensor->name, it.key().c_str(), it.key().size() + 1);
        const std::string dtype = descriptor["dtype"].get<std::string>();
        tensor->dtype = parse_dtype(dtype.c_str(), static_cast<int>(dtype.size()));
        if (tensor->dtype == SFDtype::Unknown || descriptor["shape"].size() > SF_MAX_DIMS) return false;

        uint64_t numel = 1;
        for (const auto& dimension : descriptor["shape"]) {
            if (!dimension.is_number_integer()) return false;
            int64_t value = dimension.get<int64_t>();
            if (value < 0 || (value > 0 && numel > std::numeric_limits<uint64_t>::max() / value)) {
                return false;
            }
            tensor->shape[tensor->ndims++] = value;
            numel *= static_cast<uint64_t>(value);
        }

        const auto& offsets = descriptor["data_offsets"];
        if (!offsets[0].is_number_integer() || !offsets[1].is_number_integer()) return false;
        int64_t start = offsets[0].get<int64_t>();
        int64_t end = offsets[1].get<int64_t>();
        if (start < 0 || end < start || static_cast<uint64_t>(end) > data_bytes) return false;

        const uint64_t item_size = static_cast<uint64_t>(dtype_size(tensor->dtype));
        if (item_size == 0 || (numel > 0 && item_size > std::numeric_limits<uint64_t>::max() / numel)
            || numel * item_size != static_cast<uint64_t>(end - start)) {
            return false;
        }
        tensor->data_offset = static_cast<size_t>(start);
        tensor->data_size = static_cast<size_t>(end - start);
        n_tensors_++;
    }
    return n_tensors_ > 0;
}

SafeTensors::~SafeTensors() {
    close();
}

void SafeTensors::close() {
    if (mmap_base_) {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

SafeTensors* SafeTensors::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        return nullptr;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 9) {
        ::close(fd);
        fprintf(stderr, "Invalid or truncated safetensors file: %s\n", path.c_str());
        return nullptr;
    }
    size_t file_size = st.st_size;

    void* base = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        fprintf(stderr, "mmap failed\n");
        return nullptr;
    }

    uint64_t header_size = 0;
    memcpy(&header_size, base, sizeof(header_size));
    if (header_size == 0 || header_size > file_size - 8) {
        munmap(base, file_size);
        ::close(fd);
        fprintf(stderr, "Invalid safetensors header size: %llu\n", header_size);
        return nullptr;
    }

    auto* sf = new SafeTensors();
    sf->fd_ = fd;
    sf->mmap_base_ = base;
    sf->mmap_size_ = file_size;
    sf->header_size_ = header_size;
    sf->data_base_ = (const uint8_t*)base + 8 + header_size;

    const char* json = (const char*)base + 8;
    if (!sf->parse_header(json, (int64_t)header_size)) {
        delete sf;
        fprintf(stderr, "Failed to parse safetensors header\n");
        return nullptr;
    }

    LOG("Loaded safetensors: %d tensors, %.1f MB\n", sf->n_tensors_, file_size / 1e6);
    return sf;
}

const SFTensor* SafeTensors::find(const char* name) const {
    for (int i = 0; i < n_tensors_; i++) {
        if (strcmp(tensors_[i].name, name) == 0) {
            return &tensors_[i];
        }
    }
    return nullptr;
}

const void* SafeTensors::data(const SFTensor* t) const {
    return data_base_ + t->data_offset;
}

int64_t SafeTensors::numel(const SFTensor* t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndims; i++) n *= t->shape[i];
    return n;
}

float* SafeTensors::load_bf16_to_f32(const char* name, int64_t expected_numel) const {
    const SFTensor* t = find(name);
    if (!t) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return nullptr;
    }
    int64_t n = numel(t);
    if (expected_numel > 0 && n != expected_numel) {
        fprintf(stderr, "Shape mismatch for %s: expected %lld, got %lld\n", name, expected_numel, n);
        return nullptr;
    }
    if (t->dtype != SFDtype::BF16) {
        fprintf(stderr, "Unsupported dtype for %s: expected BF16\n", name);
        return nullptr;
    }
    float* out = (float*)malloc(n * sizeof(float));
    if (!out) return nullptr;
    const uint16_t* bf16 = (const uint16_t*)data(t);
    bf16_to_f32_vec(out, bf16, (int)n);
    return out;
}

float* SafeTensors::load_f32_direct(const char* name, int64_t expected_numel) const {
    const SFTensor* t = find(name);
    if (!t) {
        fprintf(stderr, "Weight not found: %s\n", name);
        return nullptr;
    }
    int64_t n = numel(t);
    if (expected_numel > 0 && n != expected_numel) {
        fprintf(stderr, "Shape mismatch for %s: expected %lld, got %lld\n", name, expected_numel, n);
        return nullptr;
    }
    if (t->dtype != SFDtype::F32) {
        fprintf(stderr, "Unsupported dtype for %s: expected F32\n", name);
        return nullptr;
    }
    float* out = (float*)malloc(n * sizeof(float));
    if (!out) return nullptr;
    memcpy(out, data(t), n * sizeof(float));
    return out;
}

// Qwen3.5 RMSNorm uses (1+w) pattern
float* SafeTensors::load_norm_weight(const char* name, int64_t expected_numel) const {
    float* w = load_bf16_to_f32(name, expected_numel);
    if (w) {
        for (int64_t i = 0; i < expected_numel; i++) w[i] += 1.0f;
    }
    return w;
}

const uint16_t* SafeTensors::get_bf16_ptr(const char* name) const {
    const SFTensor* t = find(name);
    if (!t || t->dtype != SFDtype::BF16) return nullptr;
    return (const uint16_t*)data(t);
}

// Build ANE weight blob: 128-byte header + FP16 data
// Same format as ane_bridge_build_weight_blob / ane_runtime build_weight_blob
static bool write_ane_blob(const std::string& path, const uint16_t* bf16_data, int64_t num_elements) {
    size_t wsize = (size_t)num_elements * 2;  // FP16 = 2 bytes
    size_t total = 128 + wsize;               // header + data
    uint8_t* buf = (uint8_t*)calloc(total, 1);
    if (!buf) return false;

    // Global header (64 bytes)
    buf[0] = 0x01;
    buf[4] = 0x02;

    // Chunk header (64 bytes at offset 64)
    uint8_t* chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;  // magic
    chunk[4] = 0x01;                                                        // version
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;                              // data size
    *(uint32_t*)(chunk + 16) = 128;                                         // data offset

    // Convert BF16 → FP16 at offset 128
    uint16_t* fp16 = (uint16_t*)(buf + 128);
    bf16_to_f16_vec(fp16, bf16_data, (int)num_elements);

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { free(buf); return false; }
    fwrite(buf, 1, total, fp);
    fclose(fp);
    free(buf);
    return true;
}

int SafeTensors::write_ane_blobs(const SafeTensors& src, const std::string& output_dir) {
    int n = src.n_tensors();
    int written = 0;

    for (int i = 0; i < n; i++) {
        const SFTensor& t = src.tensor(i);
        if (t.dtype != SFDtype::BF16) continue;

        int64_t elem_count = numel(&t);
        const uint16_t* bf16_data = (const uint16_t*)src.data(&t);

        // tensor name → filename: dots to slashes for directory structure
        // e.g. "model.language_model.layers.0.mlp.gate_proj.weight"
        //    → "model/language_model/layers/0/mlp/gate_proj/weight.bin"
        std::string rel_path = t.name;
        for (char& c : rel_path) {
            if (c == '.') c = '/';
        }
        std::string full_path = output_dir + "/" + rel_path + ".bin";

        // Create parent directories
        std::string dir = full_path.substr(0, full_path.rfind('/'));
        std::string tmp;
        for (size_t j = 0; j < dir.size(); j++) {
            tmp += dir[j];
            if (dir[j] == '/' || j == dir.size() - 1)
                mkdir(tmp.c_str(), 0755);
        }

        if (!write_ane_blob(full_path, bf16_data, elem_count)) {
            fprintf(stderr, "Failed to write %s\n", full_path.c_str());
            return -1;
        }
        written++;
    }

    fprintf(stderr, "Wrote %d ANE blobs to %s\n", written, output_dir.c_str());
    return written;
}

} // namespace ane_lm
