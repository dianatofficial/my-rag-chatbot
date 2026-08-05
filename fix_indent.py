import re, pathlib, py_compile

cut = lambda s: s.split("# ===END===")[0]
conv = lambda s: re.sub(r"(?m)^\.+", lambda m: " " * len(m.group(0)), cut(s))
rd = lambda p: pathlib.Path(p).read_text(encoding="utf-8")
wr = lambda p, s: pathlib.Path(p).write_text(s, encoding="utf-8", newline="\n")

wr("rag_engine.py", conv(rd("rag_engine.txt")))
wr("build_index.py", conv(rd("build_index.txt")))
py_compile.compile("rag_engine.py", doraise=True)
py_compile.compile("build_index.py", doraise=True)
print("OK - generated and compiled")
