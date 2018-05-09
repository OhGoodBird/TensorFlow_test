fp = open('t.txt', 'w+')
fp.seek(0)
fp.write('\xef\xbb\xbf')
fp.close()
