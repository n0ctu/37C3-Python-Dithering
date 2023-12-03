# 37C3-Python-Dithering
 A more basic approach to grayscale dithering by reimplementing well known error distribution matrices in Python with Numpy.

 ## Current state and issues

 Experimental/Proof of concept. The conversion takes quite long, but the basic approach allows for control over the whole process. The Pillow library (PIL) can be used to prepare images before dithering to further influence the result.

Current challenges:
- [ ] Speed up the conversion process
- [ ] Even after adding clipping to newly calculated pixels, there still seems to be a clipping issue resulting in stray pixels populating downwards-right

![Clipping issue](issue/clipping_issue.gif)

 ## Further information

 Further information on the algorithm and distribution matrices can be found here. Great reading!
 https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html