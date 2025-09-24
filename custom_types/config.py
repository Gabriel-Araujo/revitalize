class Config:
    file_type: str
    file_path: str
    debug_mode: bool

    def __init__(self, file_type: str, file_path: str, debug_mode: bool):
        self.file_type = file_type
        self.file_path = file_path
        self.debug_mode = debug_mode

    def __str__(self) -> str:
        return format(
            f"file type: {self.file_type} \nfile path: {self.file_path} \ndebug mode: {self.debug_mode}"
        )
