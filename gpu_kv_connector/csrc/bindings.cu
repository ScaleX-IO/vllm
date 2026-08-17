#include <gpu_lsm_kvstore/object_store.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

namespace py = pybind11;
using gpu_lsm_kv::BamIo;
using gpu_lsm_kv::GpuStripedObjectStore;
using gpu_lsm_kv::ObjectDescriptor;
using gpu_lsm_kv::ObjectStoreConfig;

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

class ObjectStoreBinding {
public:
    ObjectStoreBinding(
        const std::string& device_path,
        int64_t disk_start_page,
        int64_t capacity_pages,
        int64_t plane_count,
        int64_t plane_bytes,
        int64_t max_objects,
        int64_t max_batch,
        int64_t queue_depth,
        int64_t num_queues,
        int64_t cuda_device)
        : cuda_device_(checked_u32(cuda_device, "cuda_device", true)) {
        TORCH_CHECK(disk_start_page >= 0, "disk_start_page must be non-negative");
        TORCH_CHECK(capacity_pages >= 0, "capacity_pages must be non-negative");
        c10::cuda::CUDAGuard guard(cuda_device_);
        io_ = std::make_unique<BamIo>(
            device_path.c_str(), 1, cuda_device_,
            checked_u32(queue_depth, "queue_depth"),
            checked_u32(num_queues, "num_queues"),
            static_cast<uint64_t>(disk_start_page));

        ObjectStoreConfig config;
        config.plane_count = checked_u32(plane_count, "plane_count");
        config.plane_bytes = checked_u32(plane_bytes, "plane_bytes");
        config.max_objects = checked_u32(max_objects, "max_objects");
        config.max_batch = checked_u32(max_batch, "max_batch");
        config.capacity_pages = static_cast<uint64_t>(capacity_pages);
        store_ = std::make_unique<GpuStripedObjectStore>(*io_, config);
    }

    int64_t register_region(const torch::Tensor& tensor) {
        check_cuda_contiguous(tensor, "region");
        TORCH_CHECK(tensor.get_device() == static_cast<int>(cuda_device_),
                    "region belongs to the wrong CUDA device");
        uint64_t bytes = checked_bytes(tensor);
        c10::cuda::CUDAGuard guard(cuda_device_);
        return store_->register_region(tensor.data_ptr(), bytes);
    }

    std::tuple<torch::Tensor, torch::Tensor> reserve(
        const torch::Tensor& keys, const torch::Tensor& tags) {
        check_identity_tensors(keys, tags);
        auto [descriptors, status] = allocate_result(keys.size(0), keys.options());
        c10::cuda::CUDAGuard guard(cuda_device_);
        // reserve_batch performs host-side miss compaction. Synchronize once so
        // keys produced on a non-default PyTorch stream are visible before its
        // default-stream metadata operations begin.
        C10_CUDA_CHECK(cudaDeviceSynchronize());
        store_->reserve_batch(
            reinterpret_cast<const uint64_t*>(keys.data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(tags.data_ptr<int64_t>()),
            reinterpret_cast<ObjectDescriptor*>(descriptors.data_ptr<uint8_t>()),
            status.data_ptr<uint8_t>(), static_cast<uint32_t>(keys.size(0)));
        return {descriptors, status};
    }

    std::tuple<torch::Tensor, torch::Tensor> resolve(
        const torch::Tensor& keys, const torch::Tensor& tags) {
        check_identity_tensors(keys, tags);
        auto [descriptors, status] = allocate_result(keys.size(0), keys.options());
        c10::cuda::CUDAGuard guard(cuda_device_);
        store_->resolve_async(
            reinterpret_cast<const uint64_t*>(keys.data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(tags.data_ptr<int64_t>()),
            reinterpret_cast<ObjectDescriptor*>(descriptors.data_ptr<uint8_t>()),
            status.data_ptr<uint8_t>(), static_cast<uint32_t>(keys.size(0)),
            at::cuda::getCurrentCUDAStream(cuda_device_).stream());
        return {descriptors, status};
    }

    void read_plane(
        const torch::Tensor& descriptors,
        const torch::Tensor& status,
        const torch::Tensor& offsets,
        int64_t plane,
        int64_t region_id) {
        check_io_tensors(descriptors, status, offsets);
        c10::cuda::CUDAGuard guard(cuda_device_);
        store_->read_plane_async(
            reinterpret_cast<const ObjectDescriptor*>(
                descriptors.data_ptr<uint8_t>()),
            status.data_ptr<uint8_t>(),
            reinterpret_cast<const uint64_t*>(offsets.data_ptr<int64_t>()),
            static_cast<uint32_t>(status.size(0)),
            checked_u16(plane, "plane"), checked_u32(region_id, "region_id", true),
            at::cuda::getCurrentCUDAStream(cuda_device_).stream());
    }

    void write_plane(
        const torch::Tensor& descriptors,
        const torch::Tensor& status,
        const torch::Tensor& offsets,
        int64_t plane,
        int64_t region_id) {
        check_io_tensors(descriptors, status, offsets);
        c10::cuda::CUDAGuard guard(cuda_device_);
        store_->write_plane_async(
            reinterpret_cast<const ObjectDescriptor*>(
                descriptors.data_ptr<uint8_t>()),
            status.data_ptr<uint8_t>(),
            reinterpret_cast<const uint64_t*>(offsets.data_ptr<int64_t>()),
            static_cast<uint32_t>(status.size(0)),
            checked_u16(plane, "plane"), checked_u32(region_id, "region_id", true),
            at::cuda::getCurrentCUDAStream(cuda_device_).stream());
    }

    py::dict stats() const {
        auto stats = store_->read_io_counters();
        py::dict result;
        result["write_commands"] = stats.write_commands;
        result["write_nvme_blocks"] = stats.write_nvme_blocks;
        result["read_commands"] = stats.read_commands;
        result["read_nvme_blocks"] = stats.read_nvme_blocks;
        result["allocated_objects"] = store_->allocated_objects();
        result["allocated_pages"] = store_->allocated_pages();
        result["capacity_pages"] = store_->capacity_pages();
        return result;
    }

    int64_t plane_count() const { return store_->plane_count(); }
    int64_t plane_bytes() const { return store_->plane_bytes(); }
    int64_t required_region_alignment() const {
        return store_->required_region_alignment();
    }

private:
    static uint32_t checked_u32(
        int64_t value, const char* name, bool allow_zero = false) {
        TORCH_CHECK(value >= (allow_zero ? 0 : 1) &&
                        value <= static_cast<int64_t>(UINT32_MAX),
                    name, " is out of uint32 range");
        return static_cast<uint32_t>(value);
    }

    static uint16_t checked_u16(int64_t value, const char* name) {
        TORCH_CHECK(value >= 0 && value <= static_cast<int64_t>(UINT16_MAX),
                    name, " is out of uint16 range");
        return static_cast<uint16_t>(value);
    }

    static uint64_t checked_bytes(const torch::Tensor& tensor) {
        TORCH_CHECK(tensor.numel() >= 0, "tensor size is invalid");
        return static_cast<uint64_t>(tensor.numel()) * tensor.element_size();
    }

    void check_identity_tensors(
        const torch::Tensor& keys, const torch::Tensor& tags) const {
        check_cuda_contiguous(keys, "keys");
        check_cuda_contiguous(tags, "tags");
        TORCH_CHECK(keys.scalar_type() == torch::kInt64,
                    "keys must use torch.int64");
        TORCH_CHECK(tags.scalar_type() == torch::kInt64,
                    "tags must use torch.int64");
        TORCH_CHECK(keys.dim() == 1, "keys must have shape [count]");
        TORCH_CHECK(tags.dim() == 2 && tags.size(1) == 3,
                    "tags must have shape [count, 3]");
        TORCH_CHECK(keys.size(0) == tags.size(0),
                    "keys and tags must have equal counts");
        TORCH_CHECK(keys.size(0) <= static_cast<int64_t>(UINT32_MAX),
                    "identity batch is too large");
        TORCH_CHECK(keys.get_device() == static_cast<int>(cuda_device_) &&
                        tags.get_device() == static_cast<int>(cuda_device_),
                    "identity tensors belong to the wrong CUDA device");
    }

    void check_io_tensors(
        const torch::Tensor& descriptors,
        const torch::Tensor& status,
        const torch::Tensor& offsets) const {
        check_cuda_contiguous(descriptors, "descriptors");
        check_cuda_contiguous(status, "status");
        check_cuda_contiguous(offsets, "offsets");
        TORCH_CHECK(descriptors.scalar_type() == torch::kUInt8 &&
                        status.scalar_type() == torch::kUInt8 &&
                        offsets.scalar_type() == torch::kInt64,
                    "descriptor/status/offset dtypes must be uint8/uint8/int64");
        TORCH_CHECK(descriptors.dim() == 2 &&
                        descriptors.size(1) == sizeof(ObjectDescriptor),
                    "descriptors must have shape [count, 48]");
        TORCH_CHECK(status.dim() == 1 && offsets.dim() == 1,
                    "status and offsets must have shape [count]");
        TORCH_CHECK(descriptors.size(0) == status.size(0) &&
                        status.size(0) == offsets.size(0),
                    "I/O tensor counts must match");
        TORCH_CHECK(descriptors.get_device() == static_cast<int>(cuda_device_) &&
                        status.get_device() == static_cast<int>(cuda_device_) &&
                        offsets.get_device() == static_cast<int>(cuda_device_),
                    "I/O tensors belong to the wrong CUDA device");
    }

    static std::tuple<torch::Tensor, torch::Tensor> allocate_result(
        int64_t count, const torch::TensorOptions& source_options) {
        auto byte_options = source_options.dtype(torch::kUInt8);
        return {
            torch::empty({count, static_cast<int64_t>(sizeof(ObjectDescriptor))},
                         byte_options),
            torch::empty({count}, byte_options),
        };
    }

    uint32_t cuda_device_;
    std::unique_ptr<BamIo> io_;
    std::unique_ptr<GpuStripedObjectStore> store_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<ObjectStoreBinding>(module, "ObjectStore")
        .def(py::init<const std::string&, int64_t, int64_t, int64_t, int64_t,
                      int64_t, int64_t, int64_t, int64_t, int64_t>())
        .def("register_region", &ObjectStoreBinding::register_region)
        .def("reserve", &ObjectStoreBinding::reserve)
        .def("resolve", &ObjectStoreBinding::resolve)
        .def("read_plane", &ObjectStoreBinding::read_plane)
        .def("write_plane", &ObjectStoreBinding::write_plane)
        .def("stats", &ObjectStoreBinding::stats)
        .def_property_readonly("plane_count", &ObjectStoreBinding::plane_count)
        .def_property_readonly("plane_bytes", &ObjectStoreBinding::plane_bytes)
        .def_property_readonly(
            "required_region_alignment",
            &ObjectStoreBinding::required_region_alignment);
}
