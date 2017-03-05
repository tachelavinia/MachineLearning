import matplotlib.pyplot as plt
import scipy as sp

#the web status for the last month and aggregated them in web_traffic.tsv
data = sp.genfromtxt("./data/web_traffic.tsv", delimiter="\t")

#x will contain the hours and the other, y will contain the web hits in that particular hour
x = data[:,0]
y = data[:,1]

#remove the invalid values (e.g: nan)
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

#plot the data
plt.scatter(x,y,s = 2)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])

#error will be calculated as the squared distance of the model's prediction to the real data
def error(f, x, y):
    return sp.sum((f(x)-y)**2)

# a polynomial of degree 1
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print("Model parameters: %s" % fp1)
f1 = sp.poly1d(fp1)
print(error(f1, x, y))
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
plt.plot(fx, f1(fx), linewidth=2,c="r")

# a polynomial of degree 2
f2p = sp.polyfit(x, y, 2)
f2 = sp.poly1d(f2p)
plt.plot(fx, f2(fx), linewidth=2,c="g")
print(error(f2, x, y))

# a polynomial of degree 3
f3p = sp.polyfit(x, y, 3)
f3 = sp.poly1d(f3p)
plt.plot(fx, f3(fx), linewidth=2,c="m")
print(error(f3, x, y))

# a polynomial of degree 10
f4p = sp.polyfit(x, y, 10)
f4 = sp.poly1d(f4p)
plt.plot(fx, f4(fx), linewidth=2,c="c")
print(error(f4, x, y))

# a polynomial of degree 50
f5p = sp.polyfit(x, y, 50)
f5 = sp.poly1d(f5p)
plt.plot(fx, f5(fx), linewidth=2,c="y")
plt.legend(["d=%i " % f1.order, "d=%i " % f2.order, "d=%i " % f3.order, "d=%i " % f4.order, "d=%i " % f5.order], loc="upper left")
print(error(f5, x, y))

plt.autoscale(tight=True)
plt.grid()
plt.show()

#take another look at the data
inflection = int(3.5*7*24) # calculate the inflection point in hours
xa = x[:inflection] # data before the inflection point
ya = y[:inflection]
xb = x[inflection:] # data after
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)

fxa = sp.linspace(0,xa[-1], 1000) # generate X-values for plotting
fxb = sp.linspace(xb[0],xb[-1], 1000) # generate X-values for plotting
plt.scatter(x,y,s = 2)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.plot(fxa, fa(fxa), linewidth=2,c="y")
plt.plot(fxb, fb(fxb), linewidth=2,c="r")
plt.autoscale(tight=True)
plt.grid()
plt.show()
print("Error inflection=%f" % (fa_error + fb_error))

#separate data set in training and testing data
frac = 0.3
split_idx = int(frac * len(x))
shuffled = sp.random.permutation(list(range(len(x))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(x[train], y[train], 1))
fbt2 = sp.poly1d(sp.polyfit(x[train], y[train], 2))
print("fbt2(x)= \n%s"%fbt2)
print("fbt2(x)-100,000= \n%s"%(fbt2-100000))


plt.scatter(x,y,s = 2)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])

fbt3 = sp.poly1d(sp.polyfit(x[train], y[train], 3))
plt.plot(fx, fbt3(fx), linewidth=2,c="y")
plt.autoscale(tight=True)
plt.grid()
plt.show()


