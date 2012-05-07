from pylab import *
import random

f1 = open('mu_nearest_uid_P.txt', 'r')
# f2 = open('soft_traits.txt', 'r')

lines = f1.readlines()
# y2 = f2.readlines()

# while len(y2) < len(y1):
#     y2.append(random.choice(y2))

x = range(len(lines))

y1 = []
y2 = []
y3 = []
y4 = []

for i in range(len(lines)):
    a, b, c ,d = lines[i].split()
    y1.append(a)
    y2.append(b)
    y3.append(c)

fig = figure()
ax = fig.add_subplot(111)
ax.plot(x, y1, 'b-', x, y2, 'r-', x, y3, 'g-')
leg = ax.legend(('Mu', 'Recognized user', 'Authorized user'), 'upper right', shadow=True)
ax.set_xlim([-1,250])
ax.set_ylim([0, 4])
ax.set_xlabel('Frame no. --->')
ax.set_ylabel('User ID --->')
ax.set_title('Mean vs. Recognized user vs. Authorized user')
show()

