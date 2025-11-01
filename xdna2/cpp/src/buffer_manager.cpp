#include <Python.h>  // Must come first
#include "buffer_manager.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace whisper_xdna2 {

BufferManager::BufferManager(PyObject* device_obj)
    : device_obj_(device_obj)
    , total_allocated_(0)
{
    if (!device_obj_) {
        throw std::runtime_error("device_obj cannot be null");
    }
    Py_INCREF(device_obj_);
}

BufferManager::~BufferManager() {
    clear();
    if (device_obj_) {
        Py_DECREF(device_obj_);
        device_obj_ = nullptr;
    }
}

PyObject* BufferManager::get_buffer(const std::string& name, size_t size, size_t alignment) {
    // Check if buffer already exists
    auto it = buffers_.find(name);
    if (it != buffers_.end()) {
        // Reuse existing buffer if size matches
        if (it->second.size == size) {
            return it->second.buffer;
        }
        // Otherwise, clear and recreate
        clear_buffer(name);
    }

    // Create new buffer
    // TODO: Actually create XRT buffer object
    // For now, just store nullptr as placeholder
    BufferInfo info;
    info.buffer = nullptr;  // Placeholder
    info.size = size;
    info.alignment = alignment;

    buffers_[name] = info;
    total_allocated_ += size;

    return info.buffer;
}

void BufferManager::write_buffer(const std::string& name, const void* data, size_t size) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::runtime_error("Buffer not found: " + name);
    }

    // TODO: Write to XRT buffer
    // For now, just validate
    if (size > it->second.size) {
        throw std::runtime_error("Write size exceeds buffer size");
    }
}

void BufferManager::read_buffer(const std::string& name, void* data, size_t size) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::runtime_error("Buffer not found: " + name);
    }

    // TODO: Read from XRT buffer
    // For now, just validate
    if (size > it->second.size) {
        throw std::runtime_error("Read size exceeds buffer size");
    }
}

void BufferManager::sync_to_device(const std::string& name) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::runtime_error("Buffer not found: " + name);
    }

    // TODO: Sync buffer to device
}

void BufferManager::sync_from_device(const std::string& name) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::runtime_error("Buffer not found: " + name);
    }

    // TODO: Sync buffer from device
}

bool BufferManager::has_buffer(const std::string& name) const {
    return buffers_.find(name) != buffers_.end();
}

size_t BufferManager::get_buffer_size(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        return 0;
    }
    return it->second.size;
}

void BufferManager::clear_buffer(const std::string& name) {
    auto it = buffers_.find(name);
    if (it != buffers_.end()) {
        if (it->second.buffer) {
            Py_DECREF(it->second.buffer);
        }
        total_allocated_ -= it->second.size;
        buffers_.erase(it);
    }
}

void BufferManager::clear() {
    for (auto& kv : buffers_) {
        if (kv.second.buffer) {
            Py_DECREF(kv.second.buffer);
        }
    }
    buffers_.clear();
    total_allocated_ = 0;
}

void BufferManager::call_buffer_method(PyObject* buffer, const char* method_name) {
    if (!buffer) {
        return;
    }

    PyObject* result = PyObject_CallMethod(buffer, method_name, nullptr);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error(std::string("Failed to call ") + method_name);
    }
    Py_DECREF(result);
}

} // namespace whisper_xdna2
