import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run("api.tools_main:app", host=os.environ.get("IMAGE_TOOLS_HOST", "0.0.0.0"), port=int(os.environ.get("IMAGE_TOOLS_PORT", "8091")), workers=1)
