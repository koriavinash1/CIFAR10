import os
import time

ptime = time.time()
print "running Position Based Network"
os.system('python vgg16.py 0')

ntime = time.time()
print "running Normal Network"
os.system('python vgg16.py 1')

print "---------------------------------------------------------------------------"
print "Total time taken: {}sec".format(time.time() - ptime)
print "Time taken by position based model: {}sec".format(ntime - ptime)
print "Time taken by normal network: {}sec".format(time.time() - ntime)
