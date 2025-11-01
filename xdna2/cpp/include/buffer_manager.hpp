#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <vector>

// Forward declaration - PyObject is defined in Python.h
// Include Python.h in .cpp files before this header
struct _object;
typedef struct _object PyObject;

namespace whisper_xdna2 {

/**
 * BufferManager - Efficient buffer management for NPU operations
 *
 * Manages XRT buffer objects (BOs) with:
 * - Automatic buffer pooling and reuse
 * - Zero-copy where possible
 * - Sync operations (host <-> device)
 * - Memory-efficient allocation
 *
 * Buffer Lifecycle:
 * 1. Allocate buffer on device (one-time)
 * 2. Write data to buffer (sync to device)
 * 3. Execute kernel
 * 4. Read data from buffer (sync from device)
 * 5. Reuse buffer for next operation
 */
class BufferManager {
public:
    /**
     * Construct buffer manager
     *
     * @param device_obj Python pyxrt.device object
     */
    explicit BufferManager(PyObject* device_obj);

    /**
     * Destructor - cleans up all buffers
     */
    ~BufferManager();

    // Disable copy
    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    /**
     * Get or create a buffer
     *
     * If buffer with this name and size already exists, reuse it.
     * Otherwise, allocate a new buffer.
     *
     * @param name Buffer name (e.g., "input_a", "weights", "output")
     * @param size Buffer size in bytes
     * @param alignment Buffer alignment (default: 4096 bytes for NPU)
     * @return PyObject* to the buffer object
     */
    PyObject* get_buffer(const std::string& name, size_t size, size_t alignment = 4096);

    /**
     * Write data to buffer and sync to device
     *
     * @param name Buffer name
     * @param data Pointer to source data
     * @param size Number of bytes to write
     */
    void write_buffer(const std::string& name, const void* data, size_t size);

    /**
     * Read data from buffer (sync from device)
     *
     * @param name Buffer name
     * @param data Pointer to destination buffer
     * @param size Number of bytes to read
     */
    void read_buffer(const std::string& name, void* data, size_t size);

    /**
     * Sync buffer to device (host -> device)
     *
     * @param name Buffer name
     */
    void sync_to_device(const std::string& name);

    /**
     * Sync buffer from device (device -> host)
     *
     * @param name Buffer name
     */
    void sync_from_device(const std::string& name);

    /**
     * Check if buffer exists
     */
    bool has_buffer(const std::string& name) const;

    /**
     * Get buffer size
     */
    size_t get_buffer_size(const std::string& name) const;

    /**
     * Clear specific buffer
     */
    void clear_buffer(const std::string& name);

    /**
     * Clear all buffers
     */
    void clear();

    /**
     * Get total allocated memory
     */
    size_t get_total_memory() const { return total_allocated_; }

    /**
     * Get number of buffers
     */
    size_t get_num_buffers() const { return buffers_.size(); }

private:
    struct BufferInfo {
        PyObject* buffer;      // pyxrt.bo object
        size_t size;           // Size in bytes
        size_t alignment;      // Alignment requirement
    };

    PyObject* device_obj_;     // Reference to device object
    std::unordered_map<std::string, BufferInfo> buffers_;
    size_t total_allocated_;

    // Helper to call Python methods
    void call_buffer_method(PyObject* buffer, const char* method_name);
};

/**
 * TypedBufferView - Type-safe view into a buffer
 *
 * Provides convenient typed access to buffers without copying.
 */
template<typename T>
class TypedBufferView {
public:
    TypedBufferView(BufferManager& manager, const std::string& name)
        : manager_(manager), name_(name) {}

    /**
     * Write typed data to buffer
     */
    void write(const std::vector<T>& data) {
        manager_.write_buffer(name_, data.data(), data.size() * sizeof(T));
    }

    /**
     * Write typed data from pointer
     */
    void write(const T* data, size_t count) {
        manager_.write_buffer(name_, data, count * sizeof(T));
    }

    /**
     * Read typed data from buffer
     */
    std::vector<T> read() {
        size_t buffer_size = manager_.get_buffer_size(name_);
        size_t count = buffer_size / sizeof(T);
        std::vector<T> data(count);
        manager_.read_buffer(name_, data.data(), buffer_size);
        return data;
    }

    /**
     * Read typed data into existing buffer
     */
    void read(T* data, size_t count) {
        manager_.read_buffer(name_, data, count * sizeof(T));
    }

private:
    BufferManager& manager_;
    std::string name_;
};

} // namespace whisper_xdna2
