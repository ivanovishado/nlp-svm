from chardet.universaldetector import UniversalDetector

detector = UniversalDetector()
for line in open("violent.txt", 'rb'):
    detector.feed(line)
    if detector.done:
        break
detector.close()
print(detector.result)