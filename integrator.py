import matplotlib.pyplot as plt
import numpy as np

def f1(x, y):
  dydx = x*np.sqrt(y)
  return dydx

def step01(x, y, f, h):
  y += h*f(x,y)
  return y

def stepRK4(x, y, f, h):
  
  y1 = h*f(x,y)
  y2 = h*f(x+h/2, y+y1/2)
  y3 = h*f(x+h/2, y+y2/2)
  y4 = h*f(x+h/2, y+y3)
  y = (y + (y1 + 2*y2 + 2*y3 + y4) / 6)

  return y

def ymax(y, ymax):
  if y < ymax: return True
  else: return False

def integrate(x0, y0, f, xn=None, n=None, h=None, integrator=step01, criterion=None):
  if xn is not None:
    xx = np.linspace(x0, xn, n)
    yy = np.zeros_like(xx)
    yy[0] = y0
    h = (xn-x0)/(n-1)
    for i in range(n-1):
      yy[i+1] = integrator(xx[i], yy[i], f, h)
    return xx, yy

  elif (n is not None) and (h is not None):
    xn = h*(n-1)+x0
    xx = np.linspace(x0, xn, n)
    yy = np.zeros_like(xx)
    yy[0] = y0
    for i in range(n-1):
      yy[i+1] = integrator(xx[i], yy[i], f, h)
    return xx, yy
  
  elif (h is not None) and (criterion is not None):
    x = x0
    y = y0
    xx = []
    yy = []
    while criterion(y):
      xx.append(x)
      yy.append(y)
      y = integrator(x, y, f, h)
      x += h
    xx = np.asarray(xx)
    yy = np.asarray(yy)
  return xx, yy


if __name__ == "__main__":
 
  plt.xlabel('x')
  plt.ylabel('y')
  
  #1. Integrate 1000 steps between x= 0 and x= 100
  xx_1, yy_1 = integrate(x0=0, y0=1, f=f1, xn=100, n=1000, integrator=stepRK4)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.plot(xx_1, yy_1)
  plt.show()

  #2. Integrate 2000 steps of size 0.01 from x= 0
  xx_2, yy_2 = integrate(x0=0, y0=1, f=f1, n=2000, h=0.01, integrator=stepRK4)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.plot(xx_2, yy_2)
  plt.show()

  #3. Integrate from x= 0 with step-size 0.01 until y< 100 is violated
  xx_3, yy_3 = integrate(x0=0, y0=1, f=f1, h=0.01, integrator=stepRK4, criterion=lambda y: ymax(y, 100))
  plt.xlabel('x')
  plt.ylabel('y')
  plt.plot(xx_3, yy_3)
  plt.show()