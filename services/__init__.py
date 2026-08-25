"""Request-scoped Hivision service package.

Keep this module import-light so schema and security tests do not initialize
OpenCV, Gradio, ONNX Runtime, or model weights as a side effect.
"""
