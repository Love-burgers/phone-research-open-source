"""Exercise the actual CLI under the encoding that failed on Windows CI."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('arguments,code,stream,expected', [
    (['--help'], 0, 'stdout', '启动多轮选机对话助手'),
    (['--host', '0.0.0.0'], 2, 'stderr', '远程监听必须配置 CHAT_ACCESS_TOKEN'),
])
def test_cli_chinese_output_survives_non_utf8_windows_pipe(tmp_path, arguments, code, stream, expected):
    environment = dict(os.environ, PYTHONIOENCODING='cp1252:strict', PYTHONUTF8='0',
                       PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    environment.pop('CHAT_ACCESS_TOKEN', None)
    result = subprocess.run([sys.executable, '-m', 'app.chat_cli', *arguments],
                            cwd=tmp_path, env=environment, capture_output=True, timeout=20)
    assert result.returncode == code, result.stderr.decode('utf-8', errors='replace')
    assert expected in getattr(result, stream).decode('utf-8')
    assert b'UnicodeEncodeError' not in result.stderr
