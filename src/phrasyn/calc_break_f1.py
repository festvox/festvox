#!/usr/bin/env python
###########################################################################
##                                                                       ##
##                  Language Technologies Institute                      ##
##                     Carnegie Mellon University                        ##
##                         Copyright (c) 2010                            ##
##                        All Rights Reserved.                           ##
##                                                                       ##
##  Permission is hereby granted, free of charge, to use and distribute  ##
##  this software and its documentation without restriction, including   ##
##  without limitation the rights to use, copy, modify, merge, publish,  ##
##  distribute, sublicense, and/or sell copies of this work, and to      ##
##  permit persons to whom this work is furnished to do so, subject to   ##
##  the following conditions:                                            ##
##   1. The code must retain the above copyright notice, this list of    ##
##      conditions and the following disclaimer.                         ##
##   2. Any modifications must be clearly marked as such.                ##
##   3. Original authors' names are not deleted.                         ##
##   4. The authors' names are not used to endorse or promote products   ##
##      derived from this software without specific prior written        ##
##      permission.                                                      ##
##                                                                       ##
##  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ##
##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
##  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      ##
##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
##  THIS SOFTWARE.                                                       ##
##                                                                       ##
###########################################################################
##                                                                       ##
##  Author: Alok Parlikar (aup@cs.cmu.edu)                               ##
##  Date  : July 2011                                                    ##
###########################################################################
## Description: Measures the F1 measure of predicted breaks              ##
##                                                                       ##
##                                                                       ##
###########################################################################

import sys

if len(sys.argv) != 3:
    print >>sys.stderr, "Usage: %s hypfile reffile"%sys.argv[0]
    sys.exit(-1)

taglist = ['3','1']

f = open(sys.argv[1])
g = open(sys.argv[2])

counts = {}
for i in taglist:
    counts[i] = {}
    for j in taglist:
        counts[i][j] = 0

for hyp, ref in zip(f,g):
    bhyp = hyp.split()[0]
    bref = ref.split()[0]

    if bref == '4':
        continue # Don't count BB as they are default.

    if bhyp == '4':
        bhyp = '3'

    if bref == '4':
        bref = '3'
    counts[bref][bhyp] += 1

overallcorrect = 0
overalltotal = 0

truepositive = float(counts['3']['3'])
falsepositive = float(counts['1']['3'])

truenegative = float(counts['1']['1'])
falsenegative = float(counts['3']['1'])

if truepositive == 0:
    precision = 0
    recall = 0
    fmeasure = 0
else:
    precision = truepositive/(truepositive + falsepositive)
    recall = truepositive/(truepositive+falsenegative)
    fmeasure = 2*(precision*recall)/(precision+recall)

for i in taglist:
    print "%2s"%i,
    total = 0
    correct = 0
    for j in taglist:
        total += counts[i][j]
        if i==j:
            correct = counts[i][j]
        print "%3d"%counts[i][j],
    overallcorrect += correct
    overalltotal += total
    print "\t[%d/%d]\t\t%1.2f"%(correct, total, 100.0*correct/total)

print "[%3d/%3d] = %1.2f"%(overallcorrect, overalltotal, 100.0*overallcorrect/overalltotal)
print "Prec: %1.2f Recall: %1.2f F-measure: %1.2f"%(100*precision, 100*recall, 100*fmeasure)
