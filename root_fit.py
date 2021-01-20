import numpy as np
from ROOT import TCanvas, TH1
from ROOT import TF1
#Some data
x = np.arange(10)
y = x**2

#Canvas to plot on and graph
c = TCanvas()
g = THisto

#Draw canvas and graph
c.Draw()
g.Draw()


#Use a custom function (altough the build in pol2 would also work)
func = TF1('func', '[0] + [1]*x + [2]*x**2', 0, 10)
fit = g.Fit('func', 'S')

c.Draw()
g.Draw('AP')


par = [fit.Get().Parameter(i) for i in range( 3 )]
