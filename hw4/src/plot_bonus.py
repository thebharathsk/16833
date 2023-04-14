import numpy as np
import matplotlib.pyplot as plt

data_high = np.loadtxt('./icp_loss_high_drift.txt')
data_low = np.loadtxt('./icp_loss_low_drift.txt')

plt.plot(data_high, 'r', label='High Drift')
plt.plot(data_low, 'b', label='Low Drift')

plt.xlabel('Frame number')
plt.ylabel('Loss')
plt.title('ICP Loss')
plt.legend()
plt.savefig('bonus_loss.png')
plt.close()