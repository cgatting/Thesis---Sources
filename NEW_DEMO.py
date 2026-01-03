import sys

def main(lines=None):
    if lines is None:
        lines = [line.rstrip('\n') for line in sys.stdin if line.rstrip("\n") != '']
    if not lines:
        return
    message = lines[0]
    translations = {}
    for line in lines[1:]:
        src, dst = line.split("|", 1)
        translations[src] = dst
    fragments = sorted(translations.keys(), key=len)
    results = []
    i=0
    while i < len(message):
        print(i)
        matched = False
        for frag in fragments:
            if message.startswith(frag, i):
                results.append(translations[frag])
                i += len(frag)
                matched = True
                break
        if not matched:
            results.append(message[i])
            i += 1
    print("".join(results))
   
if __name__ == "__main__":
    main(lines=[
        "hello world",
        "hello|hi",
        "world|earth"
    ])