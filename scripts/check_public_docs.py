"""Check Markdown file/anchor links against an explicit safe publication list."""
import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def anchors(text):
    counts={}; found=set()
    for line in text.splitlines():
        if not re.match(r'^#{1,6}\s+',line):continue
        heading=re.sub(r'^#+\s+','',line).strip().lower()
        heading=re.sub(r'[^\w\-\s]','',heading).replace(' ','-')
        count=counts.get(heading,0); counts[heading]=count+1
        found.add(heading+(f'-{count}' if count else ''))
    return found


def check(root, public):
    public={Path(p).as_posix() for p in public}; errors=[]; links=0
    for name in sorted(public):
        path=root/name
        if not path.is_file():
            errors.append(f'{name}: listed public file is missing');continue
        if path.suffix.lower()!='.md':continue
        text=path.read_text(encoding='utf-8')
        # Ignore fenced code; examples are not navigation links.
        text=re.sub(r'(?ms)^(`{3,}|~{3,}).*?^\1\s*$','',text)
        targets=re.findall(r'\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)',text)
        targets+=re.findall(r'(?m)^\[[^\]]+\]:\s*(\S+)',text)
        for target in targets:
            target=target.strip('<>')
            parts=urlsplit(target)
            if parts.scheme or parts.netloc:continue
            links+=1
            dest=(path.parent/unquote(parts.path)).resolve() if parts.path else path.resolve()
            try: relative=dest.relative_to(root.resolve()).as_posix()
            except ValueError:
                errors.append(f'{name}: outside public root: {target}');continue
            if relative not in public or not dest.is_file():
                errors.append(f'{name}: absent from publication list: {target}');continue
            if parts.fragment and dest.suffix.lower()=='.md' and unquote(parts.fragment).lower() not in anchors(dest.read_text(encoding='utf-8')):
                errors.append(f'{name}: missing anchor: {target}')
    return links,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('docs/public-files.txt'))
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    public=[line.strip() for line in args.manifest.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')]
    links,errors=check(root,public)
    print(f'{len(public)} files; {links} internal links; {len(errors)} errors')
    for error in errors:print(error)
    return bool(errors)


if __name__=='__main__':raise SystemExit(main())
