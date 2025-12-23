# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
A simplified multi-image shared memory tool module
When writing, concatenate three images (head, left, right) horizontally and write them
When reading, split the concatenated image into three independent images
"""

import ctypes
import time
import numpy as np
import cv2
from multiprocessing import shared_memory
from typing import Optional, Dict, List
import struct
import os

# shared memory configuration
SHM_NAME = "isaac_multi_image_shm"
SHM_SIZE = 640* 480 * 3 * 3 + 1024  # the size of the concatenated images + the header information buffer

# define the simplified header structure
class SimpleImageHeader(ctypes.Structure):
    """Simplified image header structure"""
    _fields_ = [
        ('timestamp', ctypes.c_uint64),    # timestamp
        ('height', ctypes.c_uint32),       # image height
        ('width', ctypes.c_uint32),        # total width after concatenation
        ('channels', ctypes.c_uint32),     # number of channels
        ('single_width', ctypes.c_uint32), # single image width
        ('image_count', ctypes.c_uint32),  # number of images
        ('data_size', ctypes.c_uint32),    # data size
        ('encoding', ctypes.c_uint32),     # 0=raw BGR, 1=JPEG
        ('quality', ctypes.c_uint32),      # JPEG quality (valid if encoding=1)
    ]


class MultiImageWriter:
    """A simplified multi-image shared memory writer"""
    
    def __init__(self, shm_name: str = SHM_NAME, shm_size: int = SHM_SIZE, *, enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
        """Initialize the multi-image shared memory writer
        
        Args:
            shm_name: the name of the shared memory
            shm_size: the size of the shared memory
        """
        self.shm_name = shm_name
        self.shm_size = shm_size
        
        # 50 FPS 限速（避免高频阻塞主循环）
        self._min_interval_sec = 1.0 / 50.0
        self._last_write_ts_ms = 0
        
        # 压缩与颜色空间配置（由主进程注入）
        self._enable_jpeg = bool(enable_jpeg)
        self._jpeg_quality = int(jpeg_quality)
        self._skip_cvtcolor = bool(skip_cvtcolor)
        
        try:
            # try to open the existing shared memory
            self.shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            # if not exist, create a new shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
        
        print(f"[MultiImageWriter] Shared memory initialized: {shm_name}")

    def set_options(self, *, enable_jpeg: Optional[bool] = None, jpeg_quality: Optional[int] = None, skip_cvtcolor: Optional[bool] = None):
        if enable_jpeg is not None:
            self._enable_jpeg = bool(enable_jpeg)
        if jpeg_quality is not None:
            self._jpeg_quality = int(jpeg_quality)
        if skip_cvtcolor is not None:
            self._skip_cvtcolor = bool(skip_cvtcolor)

    def write_images(self, images: Dict[str, np.ndarray]) -> bool:
        """Write multiple images to the shared memory (concatenate and write)
        
        Args:
            images: the image dictionary, the key is the image name ('head', 'left', 'right'), the value is the image array
            
        Returns:
            bool: whether the writing is successful
        """
        if not images or self.shm is None:
            return False
        
        # 轻量限速：最多 50 FPS，直接跳过多余写入，避免阻塞主循环
        now_ms = int(time.time() * 1000)
        if self._last_write_ts_ms and (now_ms - self._last_write_ts_ms) < int(self._min_interval_sec * 1000):
            return True
            
        try:
            # get the images in order: head, left, right
            frames_to_concat = []
            image_order = ['head', 'left', 'right']
            
            for image_name in image_order:
                if image_name in images:
                    image = images[image_name]
                    # 确保连续内存布局，尽量减少拷贝
                    if not image.flags['C_CONTIGUOUS']:
                        image = np.ascontiguousarray(image)
                    # OpenCV 期望 BGR 格式；可通过配置跳过转换
                    if image.ndim == 3 and image.shape[2] == 3:
                        if not self._skip_cvtcolor:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    frames_to_concat.append(image)
            
            if not frames_to_concat:
                return False
            
            # 使用 OpenCV 的 C 实现进行横向拼接（通常比 numpy 更快且更稳定）
            if len(frames_to_concat) > 1:
                concatenated_image = cv2.hconcat(frames_to_concat)
            else:
                concatenated_image = frames_to_concat[0]
            
            # get the image information
            height, total_width, channels = concatenated_image.shape
            single_width = total_width // len(frames_to_concat)

            # 准备头部
            header = SimpleImageHeader()
            header.timestamp = now_ms  # millisecond timestamp
            header.height = height
            header.width = total_width
            header.channels = channels
            header.single_width = single_width
            header.image_count = len(frames_to_concat)

            # 压缩或原始写入
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_view = memoryview(self.shm.buf)
            data_view = memoryview(self.shm.buf)

            if self._enable_jpeg:
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)]
                ok, buffer = cv2.imencode('.jpg', concatenated_image, encode_params)
                if not ok:
                    return False
                jpg_bytes = buffer.tobytes()
                header.encoding = 1
                header.quality = int(self._jpeg_quality)
                header.data_size = len(jpg_bytes)
                if header_size + header.data_size > self.shm_size:
                    print(f"[MultiImageWriter] JPEG data too large for SHM: {header.data_size} > {self.shm_size - header_size}")
                    return False
                # 写头
                header_bytes = ctypes.string_at(ctypes.byref(header), header_size)
                header_view[:header_size] = header_bytes
                # 写数据
                data_view[header_size:header_size + header.data_size] = jpg_bytes
            else:
                header.encoding = 0
                header.quality = 0
                raw_bytes = concatenated_image.tobytes()
                header.data_size = len(raw_bytes)
                if header_size + header.data_size > self.shm_size:
                    print(f"[MultiImageWriter] RAW data too large for SHM: {header.data_size} > {self.shm_size - header_size}")
                    return False
                header_bytes = ctypes.string_at(ctypes.byref(header), header_size)
                header_view[:header_size] = header_bytes
                data_view[header_size:header_size + header.data_size] = raw_bytes
            
            self._last_write_ts_ms = now_ms
            return True
            
        except Exception as e:
            print(f"shared_memory_utils [MultiImageWriter] Error writing to shared memory: {e}")
            print(f"Images: {list(images.keys())}")
            return False

    def close(self):
        """Close the shared memory"""
        if hasattr(self, 'shm') and self.shm is not None:
            self.shm.close()
            print(f"[MultiImageWriter] Shared memory closed: {self.shm_name}")


class MultiImageReader:
    """A simplified multi-image shared memory reader"""
    
    def __init__(self, shm_name: str = SHM_NAME):
        """Initialize the multi-image shared memory reader
        
        Args:
            shm_name: the name of the shared memory
        """
        self.shm_name = shm_name
        self.last_timestamp = 0
        self.buffer = {}
        
        try:
            # open the shared memory
            self.shm = shared_memory.SharedMemory(name=shm_name)
            print(f"[MultiImageReader] Shared memory opened: {shm_name}")
        except FileNotFoundError:
            print(f"[MultiImageReader] Shared memory {shm_name} not found")
            self.shm = None

    def _read_header(self) -> Optional[SimpleImageHeader]:
        if self.shm is None:
            return None
        header_size = ctypes.sizeof(SimpleImageHeader)
        header_data = bytes(self.shm.buf[:header_size])
        return SimpleImageHeader.from_buffer_copy(header_data)

    def read_images(self) -> Optional[Dict[str, np.ndarray]]:
        """Read multiple images from the shared memory (read the concatenated images and split them)
        
        Returns:
            Dict[str, np.ndarray]: the image dictionary, the key is the image name, the value is the image array; if the reading fails, return None
        """
        if self.shm is None:
            return None
            
        try:
            header = self._read_header()
            if header is None:
                return None
            # check if there is new data
            if header.timestamp <= self.last_timestamp:
                return self.buffer
                
            # read the payload
            header_size = ctypes.sizeof(SimpleImageHeader)
            start_offset = header_size
            end_offset = start_offset + header.data_size
            payload = bytes(self.shm.buf[start_offset:end_offset])

            # decode if needed
            if header.encoding == 1:  # JPEG
                encoded = np.frombuffer(payload, dtype=np.uint8)
                concatenated_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if concatenated_image is None:
                    return None
            else:  # RAW
                concatenated_image = np.frombuffer(payload, dtype=np.uint8)
                expected_size = header.height * header.width * header.channels
                if concatenated_image.size != expected_size:
                    print(f"[MultiImageReader] Data size mismatch: expected {expected_size}, got {concatenated_image.size}")
                    return None
                concatenated_image = concatenated_image.reshape(header.height, header.width, header.channels)
            
            # split the images
            images = {}
            image_names = ['head', 'left', 'right']
            single_width = header.single_width
            
            for i in range(header.image_count):
                if i < len(image_names):
                    start_col = i * single_width
                    end_col = start_col + single_width
                    single_image = concatenated_image[:, start_col:end_col, :]
                    images[image_names[i]] = single_image
            
            # update the buffer and timestamp
            self.buffer = images
            self.last_timestamp = header.timestamp
            return images
            
        except Exception as e:
            print(f"[MultiImageReader] Error reading from shared memory: {e}")
            return None

    def read_concatenated_image(self) -> Optional[np.ndarray]:
        """Read the concatenated image (without splitting)
        
        Returns:
            np.ndarray: the concatenated image array; if the reading fails, return None
        """
        if self.shm is None:
            return None
            
        try:
            header = self._read_header()
            if header is None:
                return None
            if header.timestamp <= self.last_timestamp:
                return None
            header_size = ctypes.sizeof(SimpleImageHeader)
            start_offset = header_size
            end_offset = start_offset + header.data_size
            payload = bytes(self.shm.buf[start_offset:end_offset])

            if header.encoding == 1:
                encoded = np.frombuffer(payload, dtype=np.uint8)
                concatenated_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if concatenated_image is None:
                    return None
            else:
                concatenated_image = np.frombuffer(payload, dtype=np.uint8)
                expected_size = header.height * header.width * header.channels
                if concatenated_image.size != expected_size:
                    print(f"[MultiImageReader] Data size mismatch: expected {expected_size}, got {concatenated_image.size}")
                    return None
                concatenated_image = concatenated_image.reshape(header.height, header.width, header.channels)
            
            self.last_timestamp = header.timestamp
            return concatenated_image
            
        except Exception as e:
            print(f"[MultiImageReader] Error reading concatenated image from shared memory: {e}")
            return None

    def read_encoded_frame(self) -> Optional[bytes]:
        """Read encoded payload if available (e.g., JPEG). Returns bytes or None."""
        if self.shm is None:
            return None
        try:
            header = self._read_header()
            if header is None or header.encoding != 1:
                return None
            if header.timestamp <= self.last_timestamp:
                return None
            header_size = ctypes.sizeof(SimpleImageHeader)
            start_offset = header_size
            end_offset = start_offset + header.data_size
            payload = bytes(self.shm.buf[start_offset:end_offset])
            self.last_timestamp = header.timestamp
            return payload
        except Exception as e:
            print(f"[MultiImageReader] Error reading encoded frame: {e}")
            return None

    def close(self):
        """Close the shared memory"""
        if self.shm is not None:
            self.shm.close()
            print(f"[MultiImageReader] Shared memory closed: {self.shm_name}")


# backward compatible class (single image)
class SharedMemoryWriter:
    """Backward compatible single image writer"""
    
    def __init__(self, shm_name: str = SHM_NAME, shm_size: int = SHM_SIZE):
        self.multi_writer = MultiImageWriter(shm_name, shm_size)
    
    def write_image(self, image: np.ndarray) -> bool:
        """Write a single image (as the head image)"""
        return self.multi_writer.write_images({'head': image})
    
    def close(self):
        self.multi_writer.close()


class SharedMemoryReader:
    """Backward compatible single image reader"""
    
    def __init__(self, shm_name: str = SHM_NAME):
        self.multi_reader = MultiImageReader(shm_name)
    
    def read_image(self) -> Optional[np.ndarray]:
        """Read a single image (the head image)"""
        images = self.multi_reader.read_images()
        return images.get('head') if images else None
    
    def close(self):
        self.multi_reader.close() 