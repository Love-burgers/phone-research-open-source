"""Start the standalone conversation service."""
import argparse
import sys
import uvicorn
from .config import get_settings


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="启动多轮选机对话助手")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8130)
    args = parser.parse_args(argv)
    if args.host not in {'127.0.0.1','localhost','::1'} and not get_settings().chat_access_token:
        parser.error('远程监听必须配置 CHAT_ACCESS_TOKEN。')
    uvicorn.run("app.web:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
