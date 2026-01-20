"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
Auto-generated from: /home/unmars/Downloads/RL_Sailing_Challenge/src/agents/relative_agent.py
"""

import numpy as np
from agents.base_agent import BaseAgent
from src.sailing_physics import calculate_sailing_efficiency

class MyAgentTrained(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()

        self.learning_rate = 0.2         # alpha - start high
        self.min_learning_rate = 0.05    # minimum alpha
        self.lr_decay_rate = 0.999       # decay per episode
        self.discount_factor = 0.99  # gamma
        self.exploration_rate = 0.3      # Start high
        self.min_exploration = 0.01      # Minimum epsilon
        self.eps_decay_rate = 0.995      # Decay per episode
        self.angle_bins = 8
        self.speed_bins = 3
        self.goal_position = np.array([16, 31])
        self.q_init_high = 10.0
        self.last_efficiency = 0.0
        self.last_vmg = 0.0
        self.prev_vmg = 0.0 # VMG of the action taken in the PREVIOUS step
        self.mask_threshold = 0.4

        # Q-table with learned values
        self.q_table = {}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        self.q_table[(2, 0, 0)] = np.array([220.8242, 110.7629,   8.586 , 182.893 ,  98.3657, 187.5817, 159.9211,
 211.7418,   1.2811])
        self.q_table[(2, 3, 2)] = np.array([376.2902, 331.8453, 152.3085, 141.7872, 224.2286, 295.777 , 299.5564,
 284.4128,   8.2763])
        self.q_table[(2, 4, 1)] = np.array([394.8039, 324.566 , 161.9786, 220.2008, 279.5603, 196.1387, 266.8732,
 301.9052,   1.5429])
        self.q_table[(2, 5, 1)] = np.array([334.4588, 215.6197,  72.4991, 132.771 , 175.2839, 215.2582, 261.4116,
 222.7339,   2.2691])
        self.q_table[(2, 7, 1)] = np.array([400.9467, 371.8459,   7.0027,  99.5656, 276.7505, 301.1336, 339.9498,
 332.1166,   6.825 ])
        self.q_table[(2, 2, 1)] = np.array([353.8837, 421.2122, 320.3925, 162.5847, 282.6182, 274.0739, 286.461 ,
 283.2708,   1.1453])
        self.q_table[(2, 1, 2)] = np.array([3.6101e+02, 3.8970e+02, 3.7266e+02, 2.3549e+02, 2.6586e+02, 3.3539e+02,
 3.3714e+02, 3.0755e+02, 2.2712e-01])
        self.q_table[(2, 3, 1)] = np.array([288.5996, 328.2981, 179.0368, 161.4397, 287.8326, 279.6373, 179.9814,
 285.6575,   3.6181])
        self.q_table[(2, 4, 2)] = np.array([304.9454, 191.0673,  73.2353,  38.9035,  55.7637,  91.4128, 172.4424,
 244.0108,   0.9639])
        self.q_table[(2, 0, 1)] = np.array([360.8763, 403.3069, 292.5316, 197.6968, 340.691 , 291.9281, 307.4167,
 372.9022,   5.841 ])
        self.q_table[(3, 0, 1)] = np.array([169.9218, 187.1476, 185.8171, 148.039 , 178.9087, 177.0796, 258.1177,
 203.9776,   2.6698])
        self.q_table[(3, 2, 1)] = np.array([203.6709, 228.9425, 165.7844, 179.4765, 189.0293, 298.2771, 191.0658,
 183.3566,   6.7224])
        self.q_table[(3, 1, 2)] = np.array([197.5158, 208.1231, 222.96  , 183.7163, 188.8364, 162.6377, 176.428 ,
 198.4875,   3.7092])
        self.q_table[(3, 3, 1)] = np.array([224.5793, 150.1908, 174.2663, 182.7904, 235.5803, 163.4122, 159.6245,
 167.6968,   0.4491])
        self.q_table[(3, 5, 2)] = np.array([233.3641, 242.361 , 177.4113, 195.0988, 207.491 , 311.9956, 180.4479,
 153.9582,   7.4679])
        self.q_table[(3, 7, 1)] = np.array([141.4385, 183.4211, 184.9547, 196.4283, 237.163 , 185.8986, 153.15  ,
 150.4861,   3.1032])
        self.q_table[(3, 4, 2)] = np.array([181.1111, 151.6509, 162.7782, 120.3529, 144.9465, 176.4415, 229.0148,
 157.3418,   5.8112])
        self.q_table[(3, 1, 1)] = np.array([237.2262, 200.639 , 189.8204, 230.4207, 314.0856, 247.9145, 177.9463,
 179.9971,   4.8858])
        self.q_table[(3, 2, 2)] = np.array([226.4108, 242.1537, 193.0744, 167.1742, 155.6336, 169.0471, 178.2999,
 192.1657,   1.0914])
        self.q_table[(3, 5, 1)] = np.array([204.8035, 210.6466, 193.3478, 166.8651, 199.9338, 323.2481, 248.442 ,
 171.8766,   4.2588])
        self.q_table[(4, 4, 1)] = np.array([338.4639, 335.7908, 316.2924, 324.3223, 314.7899, 331.3717, 341.7748,
 328.2428,   5.9443])
        self.q_table[(3, 6, 1)] = np.array([2.3240e+02, 2.0748e+02, 2.1235e+02, 2.2377e+02, 2.5857e+02, 3.3608e+02,
 2.2111e+02, 1.6164e+02, 2.1148e-01])
        self.q_table[(3, 4, 1)] = np.array([266.9054, 129.1108, 169.4071, 199.2616, 198.6846, 232.7137, 226.7143,
 181.5573,   2.3674])
        self.q_table[(4, 1, 2)] = np.array([199.0907, 207.4509, 215.6935, 221.9225, 330.9594, 284.0594, 142.4323,
 181.1989,   6.3956])
        self.q_table[(4, 3, 1)] = np.array([342.3572, 345.1577, 325.4374, 327.1239, 327.3058, 317.1167, 332.5974,
 361.0244,   9.4886])
        self.q_table[(4, 2, 1)] = np.array([350.0062, 338.5747, 329.2063, 319.9163, 325.6777, 326.4578, 333.0338,
 339.8586,   2.8816])
        self.q_table[(4, 5, 2)] = np.array([323.0432, 330.897 , 316.0008, 309.5372, 316.3023, 357.9072, 336.4211,
 360.6417,   9.3594])
        self.q_table[(4, 3, 2)] = np.array([337.9266, 346.4285, 311.2165, 309.6762, 323.3979, 319.598 , 317.8085,
 349.4571,   6.1617])
        self.q_table[(4, 6, 1)] = np.array([344.0959, 325.0122, 330.8551, 320.4672, 323.4628, 339.1706, 347.3593,
 368.9582,   1.1973])
        self.q_table[(4, 6, 2)] = np.array([342.7287, 343.1619, 323.9108, 330.7495, 333.4421, 321.648 , 335.8284,
 400.8221,   1.5761])
        self.q_table[(4, 7, 1)] = np.array([357.7535, 354.4224, 329.7288, 331.8588, 318.6324, 325.2913, 337.0388,
 388.541 ,   8.5053])
        self.q_table[(4, 5, 1)] = np.array([351.8034, 325.2018, 316.6625, 321.1471, 329.0605, 326.7914, 336.9776,
 360.1897,   1.1059])
        self.q_table[(5, 3, 2)] = np.array([371.3093, 378.7118, 361.1963, 344.7464, 354.8197, 342.1097, 345.7779,
 357.2596,   1.4302])
        self.q_table[(5, 2, 1)] = np.array([395.2377, 376.2783, 363.8221, 355.3516, 348.5637, 349.4954, 356.3858,
 427.847 ,   8.4941])
        self.q_table[(5, 7, 1)] = np.array([388.5052, 397.7453, 370.3018, 375.4937, 378.5587, 368.4013, 372.0169,
 417.7935,   6.3001])
        self.q_table[(5, 6, 1)] = np.array([401.2547, 393.2483, 381.462 , 346.834 , 379.2745, 352.5301, 355.3576,
 436.7216,   6.1631])
        self.q_table[(5, 3, 1)] = np.array([397.5962, 374.4985, 378.6116, 353.6663, 358.6639, 357.8901, 342.418 ,
 428.7456,   6.3641])
        self.q_table[(5, 5, 1)] = np.array([392.603 , 382.9535, 365.3567, 358.134 , 371.2179, 358.1873, 371.3071,
 417.8959,   4.7417])
        self.q_table[(5, 1, 1)] = np.array([3.9440e+02, 3.9828e+02, 3.6087e+02, 3.4539e+02, 3.6333e+02, 3.4163e+02,
 3.2455e+02, 4.1745e+02, 3.6060e-01])
        self.q_table[(5, 7, 2)] = np.array([399.3915, 391.4747, 393.2906, 382.8175, 387.7052, 352.6135, 397.7474,
 443.0421,   5.4914])
        self.q_table[(5, 4, 1)] = np.array([393.5303, 392.5046, 373.0611, 352.7521, 366.6499, 351.6884, 343.198 ,
 415.9506,   0.5557])
        self.q_table[(5, 6, 2)] = np.array([404.7913, 394.3541, 386.0078, 359.8627, 372.9455, 369.9715, 355.0666,
 427.0902,   3.1096])
        self.q_table[(6, 6, 2)] = np.array([393.5828, 285.9748, 251.196 , 247.3279, 241.6212,  35.8684, 206.9347,
 385.3393,   4.3378])
        self.q_table[(6, 6, 1)] = np.array([387.5045, 362.3203, 353.3911, 270.1345, 329.1611, 234.6883, 356.9571,
 437.9271,   4.9329])
        self.q_table[(6, 0, 1)] = np.array([391.1458, 378.932 , 373.8698, 341.7861, 317.3089, 204.8599, 370.4265,
 415.4627,   5.3953])
        self.q_table[(4, 7, 2)] = np.array([368.8123, 311.8891, 342.2696, 322.5513, 327.2126, 333.9213, 350.2146,
 369.2866,   7.207 ])
        self.q_table[(5, 4, 2)] = np.array([395.3229, 388.4732, 333.2545, 314.2096, 343.899 , 278.0529, 257.9131,
 386.0605,   9.8458])
        self.q_table[(5, 5, 2)] = np.array([403.2982, 347.8826, 362.4512, 295.8309, 330.0267, 312.0651, 270.4341,
 427.0868,   4.1158])
        self.q_table[(5, 0, 1)] = np.array([399.9717, 404.195 , 375.0818, 367.0322, 337.7073, 360.8748, 349.6092,
 421.4179,   9.9758])
        self.q_table[(6, 4, 1)] = np.array([390.2185, 361.7216, 300.0244, 252.1745, 291.306 , 118.5177, 222.8314,
 403.4002,   2.4137])
        self.q_table[(6, 4, 2)] = np.array([193.9459, 158.5428, 230.0818, 219.53  , 159.786 ,  84.9021, 124.2606,
 377.614 ,   9.947 ])
        self.q_table[(6, 5, 2)] = np.array([264.9822, 385.9233, 269.1914, 261.5922, 101.0277, 191.8475, 245.8142,
 373.3409,   5.1273])
        self.q_table[(6, 7, 2)] = np.array([382.0301, 377.9262, 386.9809, 334.3058, 356.0483, 318.6415, 411.1046,
 392.0153,   3.1257])
        self.q_table[(6, 3, 1)] = np.array([296.8135, 341.9442, 238.775 , 316.2473, 252.9194, 109.5998, 183.564 ,
 373.9158,   8.6276])
        self.q_table[(6, 5, 1)] = np.array([421.6216, 366.4667, 283.7482, 320.7713, 204.2424, 223.753 , 243.7158,
 326.2535,   6.8326])
        self.q_table[(6, 7, 1)] = np.array([430.2857, 374.629 , 344.6305, 337.6938, 257.8145, 229.5197, 291.6024,
 335.8635,   8.5337])
        self.q_table[(6, 0, 2)] = np.array([387.481 , 410.8474, 386.3481, 324.5401, 323.4003, 255.0744, 236.1941,
 347.1539,   3.6154])
        self.q_table[(4, 4, 2)] = np.array([2.4114e+02, 3.1872e+02, 2.8066e+02, 1.7636e+02, 2.7715e+02, 2.9675e+02,
 1.7376e+02, 1.7743e+02, 1.2470e-01])
        self.q_table[(4, 2, 2)] = np.array([340.9889, 335.4893, 318.5768, 317.2376, 329.2052, 329.987 , 301.594 ,
 279.4115,   7.3114])
        self.q_table[(5, 2, 2)] = np.array([321.899 , 217.1958, 264.1413, 344.0443, 258.6403, 262.0676, 154.7573,
 163.1314,   8.8757])
        self.q_table[(5, 0, 2)] = np.array([401.127 , 397.7152, 382.2226, 372.4332, 313.1955, 295.4672, 136.4503,
 415.2719,   9.8457])
        self.q_table[(4, 1, 1)] = np.array([345.3691, 346.585 , 321.2616, 325.0832, 316.419 , 338.1542, 324.4713,
 315.4176,   2.6454])
        self.q_table[(4, 0, 1)] = np.array([334.9818, 319.229 , 346.9541, 315.9007, 310.7074, 310.0931, 330.8149,
 324.8501,   7.7736])
        self.q_table[(6, 1, 1)] = np.array([363.116 , 284.6215, 167.7428, 198.6486, 104.5185, 125.0584,  67.4232,
 208.4538,   4.611 ])
        self.q_table[(6, 1, 2)] = np.array([373.8447,  77.6402,  75.8613,  28.2699,  89.7475,  19.9983,  86.4655,
  74.8911,   9.9181])
        self.q_table[(3, 0, 0)] = np.array([177.4107,   6.8616,   6.5269, 141.9698, 138.5449, 186.6333, 185.7822,
 219.1593,   9.7666])
        self.q_table[(6, 3, 2)] = np.array([117.7769, 327.5672,  44.035 ,   5.9695,  28.5891, 144.2442,  89.1618,
  35.7369,   3.5867])
        self.q_table[(6, 2, 1)] = np.array([255.6045, 406.5933, 254.7782, 199.7279, 200.8544, 103.1764,  30.5934,
  55.5401,   0.9464])
        self.q_table[(7, 1, 2)] = np.array([262.3953, 186.726 , 161.0063, 116.7121,  14.9241,  65.0001, 171.9633,
 332.5692,   4.6576])
        self.q_table[(7, 7, 1)] = np.array([194.4783, 314.5387, 172.8067,  62.3129,  57.6811, 289.6701, 169.8959,
 188.0527,   5.3649])
        self.q_table[(7, 1, 1)] = np.array([284.4824, 125.7612,  97.7154,  28.4779,  55.0302,  87.0561, 151.9224,
 222.7496,   4.2498])
        self.q_table[(7, 2, 1)] = np.array([ 94.8521, 245.8274,  27.4574,  30.3498,   0.8143,   6.6833,  59.2072,
  40.918 ,   0.401 ])
        self.q_table[(7, 5, 1)] = np.array([221.0083, 107.6958,  21.2272,  67.4229,   3.4414,  65.6747,  82.7539,
  55.6375,   8.0897])
        self.q_table[(7, 0, 1)] = np.array([120.7081, 138.3772, 138.4848,  46.7372,   4.2903,  84.5274, 195.8236,
 291.9837,   9.9354])
        self.q_table[(7, 6, 1)] = np.array([104.2264, 160.8053, 150.5526,  58.4153,  61.0562,  93.9718, 202.542 ,
 281.6448,   0.5982])
        self.q_table[(3, 3, 2)] = np.array([222.4837, 144.9938,  49.8526, 206.3168, 141.8636,  96.1924,  89.6686,
 113.4937,   9.7574])
        self.q_table[(3, 6, 2)] = np.array([ 47.1594, 232.867 , 207.9   , 141.5106, 124.9023, 321.9956, 191.5422,
  73.2606,   6.3091])
        self.q_table[(2, 1, 1)] = np.array([367.9397, 399.827 , 296.5075, 235.7815, 306.3268, 341.6264, 313.6636,
 306.9595,   5.2198])
        self.q_table[(3, 0, 2)] = np.array([271.2797,  91.7342,   1.7217, 157.9238, 156.3993, 163.2324, 192.5183,
 181.6105,   3.8094])
        self.q_table[(2, 0, 2)] = np.array([367.5104, 370.8783, 305.7035, 220.3278, 295.0585, 276.3339, 345.113 ,
 364.0803,   9.3555])
        self.q_table[(2, 6, 1)] = np.array([324.9738,   8.2865,  51.7348, 112.4475,  61.078 ,   2.5933, 159.6269,
 110.374 ,   4.9709])
        self.q_table[(6, 3, 0)] = np.array([ 5.8405,  4.2057,  0.1519,  0.6984,  1.9528, 31.5449, 91.5762,  5.4564,
  1.0557])
        self.q_table[(4, 7, 0)] = np.array([ 4.9209,  0.835 ,  8.3711,  1.7034,  4.1062, 41.4152,  4.5778, 60.2438,
  2.0931])
        self.q_table[(2, 2, 2)] = np.array([123.1115, 384.8842,  64.0506,   6.1605,  41.9235, 273.4389, 179.7391,
 246.5327,   8.5507])
        self.q_table[(2, 5, 2)] = np.array([ 46.1239,  20.6447,   0.5786,  39.4784,   4.1277, 172.3564,   3.9175,
 110.0397,   5.3646])
        self.q_table[(7, 4, 2)] = np.array([ 6.1072,  7.5032,  7.1466, -1.6257,  2.087 ,  3.0483, 56.1477,  4.0686,
  3.0179])
        self.q_table[(7, 4, 1)] = np.array([  2.77  ,  71.0428,  92.1326,   0.3973,   7.311 ,   3.2499, 224.211 ,
  83.3834,   1.3397])
        self.q_table[(7, 5, 2)] = np.array([ 28.6898,   4.8641,  74.9717,  17.0751,   8.042 ,  55.766 , 184.1631,
  47.8623,   7.576 ])
        self.q_table[(0, 5, 2)] = np.array([  3.1865,  31.6436,   6.0795,   2.2577,   7.809 ,   6.7988,   7.7319,
 306.2466,   5.8455])
        self.q_table[(0, 7, 2)] = np.array([444.4744, 371.9687, 372.1913, 340.1533, 260.8413, 356.9958, 395.5043,
 409.5423,   0.6289])
        self.q_table[(0, 2, 1)] = np.array([380.4188, 323.2315, 393.0955, 206.568 , 102.5592, 164.061 , 359.2435,
 353.5818,   4.0952])
        self.q_table[(0, 6, 1)] = np.array([349.8532, 396.2495, 294.5092, 228.5117, 129.9004, 276.2227, 248.7556,
 446.1563,   4.3832])
        self.q_table[(0, 1, 2)] = np.array([347.9396, 373.2113, 361.4701, 379.7563,  81.5542, 369.8268, 416.9793,
 344.0084,   3.0303])
        self.q_table[(7, 7, 2)] = np.array([172.4884, 132.8685, 165.5345,  11.5332,  20.6933, 167.4206, 231.3252,
 319.0677,   4.3093])
        self.q_table[(4, 2, 0)] = np.array([ 2.6742,  5.6896,  0.082 ,  3.7445, 38.8304,  4.6301,  4.9926,  6.2158,
  4.3421])
        self.q_table[(5, 1, 0)] = np.array([ 42.9918,   4.2716, 176.346 ,   2.6587,   2.3371,   3.1983,   4.8054,
   2.0988,   6.8421])
        self.q_table[(0, 3, 2)] = np.array([  8.2458,   8.7396,   5.2835,   2.286 ,   6.611 ,   1.4159,  84.6135,
 201.7496,   2.6649])
        self.q_table[(1, 3, 2)] = np.array([437.7626, 313.3655, 158.3839,   8.1014, 180.2557, 281.922 , 311.5387,
 305.8132,   2.3065])
        self.q_table[(4, 0, 0)] = np.array([  7.6906, 351.4576, 343.1926, 306.2458, 312.8745, 316.6826, 317.3874,
 314.6906,   9.8112])
        self.q_table[(0, 1, 1)] = np.array([386.9342, 373.8989, 388.5631, 353.514 ,  83.904 , 239.3435, 366.2088,
 427.8948,   9.0311])
        self.q_table[(7, 6, 2)] = np.array([138.7318,  40.9574,  63.0463,  70.5319,   5.9889, 101.229 , 125.9709,
 253.646 ,   6.0309])
        self.q_table[(7, 0, 2)] = np.array([124.396 , 220.7865, 124.6262,  82.3674,  36.0403,   0.9578, 158.0122,
 320.4373,   6.597 ])
        self.q_table[(5, 7, 0)] = np.array([ 5.6386, 57.0538,  2.7412, 38.3082,  4.9567,  4.9589,  5.9391,  6.1212,
  6.169 ])
        self.q_table[(0, 2, 2)] = np.array([411.7888, 382.7129, 370.0326, 325.1759,   5.5562,   2.8733, 285.8708,
 361.7864,   2.7087])
        self.q_table[(0, 7, 1)] = np.array([383.8887, 362.9792, 373.1218, 322.1905, 107.2717, 201.7891, 383.3414,
 434.1505,   2.2483])
        self.q_table[(0, 0, 1)] = np.array([334.184 , 402.3148, 349.3225, 325.1635,   5.4074, 353.4442, 428.1551,
 433.0395,   6.6879])
        self.q_table[(1, 2, 2)] = np.array([429.8371, 430.1907, 392.3452, 270.7509,   8.3833, 255.2223, 315.1707,
 396.7125,   4.1771])
        self.q_table[(1, 6, 1)] = np.array([244.6248, 318.6348, 125.5097, 123.1298,  87.7414, 219.789 , 294.8976,
 445.2494,   6.473 ])
        self.q_table[(1, 3, 1)] = np.array([437.2265, 427.5323, 355.9841, 187.4698, 164.5045, 341.1724, 389.457 ,
 365.1826,   6.2979])
        self.q_table[(1, 0, 2)] = np.array([426.8316, 422.8347, 469.1098, 371.3405, 369.4643, 401.468 , 385.5301,
 424.0018,   9.314 ])
        self.q_table[(0, 6, 2)] = np.array([224.8412, 186.7432, 141.5779,  72.2326,  30.8229,  53.6582, 249.0907,
 448.9959,   8.0608])
        self.q_table[(0, 5, 1)] = np.array([253.6881, 360.2795, 229.916 , 244.1479,  -1.2916, 119.4681, 211.2676,
 296.4976,   7.0305])
        self.q_table[(2, 7, 2)] = np.array([395.3131, 324.2831,   5.7669, 106.5056, 210.3237, 292.4013, 288.7996,
 346.5493,   5.5412])
        self.q_table[(5, 3, 0)] = np.array([ 1.5256,  2.4181,  5.5067,  1.4191, 28.0422,  9.6771,  2.6644,  6.4445,
  4.1023])
        self.q_table[(2, 5, 0)] = np.array([ 0.8236, 71.8592,  6.9578,  7.1012,  7.3426,  5.4059, 28.5656,  5.0319,
  2.016 ])
        self.q_table[(4, 1, 0)] = np.array([  2.5472, 219.6785,   4.4011,  65.7224,   5.0962,   2.4371,   1.5917,
   1.752 ,   6.7494])
        self.q_table[(5, 2, 0)] = np.array([199.635 ,   1.4815,  35.4959,   5.315 ,  57.4886,  55.3274,   4.085 ,
  34.6049,   6.798 ])
        self.q_table[(1, 1, 1)] = np.array([4.2609e+02, 4.2413e+02, 4.3452e+02, 3.8273e+02, 2.9876e+02, 3.8091e+02,
 3.6274e+02, 4.2425e+02, 4.3357e-01])
        self.q_table[(1, 1, 2)] = np.array([447.0966, 417.0722, 413.5206, 143.7984, 277.2309, 375.9934, 406.3069,
 341.958 ,   3.4748])
        self.q_table[(1, 2, 1)] = np.array([385.4372, 434.2085, 390.5278, 227.13  , 194.8708, 351.3596, 376.7142,
 374.5289,   4.0836])
        self.q_table[(3, 7, 2)] = np.array([  7.4336,   3.9564,  50.6791,  60.546 ,   2.4818, 100.4755,   1.2138,
 140.3655,   9.2998])
        self.q_table[(7, 3, 1)] = np.array([199.9895,  44.625 ,   7.1167,  72.2098,   8.2045,   3.0354,  13.5563,
   0.6662,   8.1979])
        self.q_table[(3, 6, 0)] = np.array([ 0.3185,  9.2855,  1.8322,  1.1431, 70.789 , 59.2197, 35.814 ,  0.9562,
  0.1569])
        self.q_table[(1, 0, 0)] = np.array([4.4337e+02, 3.8204e+02, 1.1593e+02, 1.3790e+00, 2.1965e+02, 3.5915e+02,
 3.8620e+02, 4.1900e+02, 1.1046e-01])
        self.q_table[(1, 0, 1)] = np.array([460.3242, 433.0155, 385.4196, 210.5666, 351.1317, 387.388 , 395.5303,
 432.5344,   3.2552])
        self.q_table[(1, 4, 1)] = np.array([441.3655, 335.2058, 150.7676,  63.3585, 272.6431, 326.0375, 278.8692,
 388.7831,   5.7837])
        self.q_table[(1, 7, 2)] = np.array([436.9041, 431.0747, 411.1997, 210.5584, 351.5158, 408.4934, 429.8328,
 429.9938,   8.9497])
        self.q_table[(0, 3, 1)] = np.array([160.3815, 236.8219, 177.4063, 162.3158,   4.479 , 247.0656, 366.901 ,
 242.938 ,   5.9904])
        self.q_table[(1, 7, 1)] = np.array([451.4235, 409.5807, 376.3613, 109.5757, 285.4778, 362.9995, 391.9723,
 409.5602,   4.3784])
        self.q_table[(0, 0, 0)] = np.array([331.4678, 366.1565, 303.3652, 203.5661,   9.7823, 224.604 , 320.1341,
 435.7085,   2.5775])
        self.q_table[(1, 5, 1)] = np.array([257.1942, 431.6998,  47.4493,   7.7722,  70.9135,  80.9775, 153.935 ,
 329.1177,   6.5305])
        self.q_table[(0, 0, 2)] = np.array([362.6816, 370.3005, 359.3698, 347.6626,  18.5253, 195.375 , 381.0436,
 426.2786,   2.8783])
        self.q_table[(7, 0, 0)] = np.array([245.76  , 201.2239, 159.7408, 113.8228,   5.832 ,   5.7998, 176.2836,
 328.6721,   3.1457])
        self.q_table[(1, 6, 2)] = np.array([ 77.0436,  72.3719,   6.2053,   5.459 , 101.1587, 132.9812, 371.9139,
 200.6098,   1.9219])
        self.q_table[(1, 6, 0)] = np.array([2.3253, 4.365 , 2.5404, 3.7951, 8.5094, 8.7324, 7.646 , 7.1148, 2.6619])
        self.q_table[(0, 4, 1)] = np.array([ 89.5965,   1.6494,   8.2435,   0.7988,   4.7957, 179.8082,   3.4595,
   1.0629,   9.9353])
        self.q_table[(5, 1, 2)] = np.array([ 6.2877,  0.4221,  1.9606,  1.9766, 42.7953,  9.9185,  8.6792,  2.5765,
  0.5611])
        self.q_table[(1, 4, 2)] = np.array([  9.5384,   5.5281, 187.7753,   9.9587,   5.1222,   3.6903,   0.9035,
   3.8201,   5.547 ])
        self.q_table[(0, 5, 0)] = np.array([ 0.9915, 69.3876,  9.2777,  1.9464,  8.7939,  1.8947,  9.6298,  1.8881,
  5.4167])
        self.q_table[(0, 6, 0)] = np.array([ 5.247 , 36.8336,  4.957 ,  7.4757,  3.805 ,  6.3094,  4.6874,  9.4346,
  7.1802])
        self.q_table[(1, 4, 0)] = np.array([  8.3092, 113.0636,   2.1366,   0.609 ,   4.9704,   1.1472,   4.1547,
 298.9128,   4.2164])
        self.q_table[(5, 0, 0)] = np.array([  7.879 , 405.302 , 343.6032, 307.4674, 320.9172, 305.4112, 340.5039,
   4.2323,   1.0791])
        self.q_table[(7, 4, 0)] = np.array([ 5.3223, 35.8983,  8.6291,  4.6454,  4.9774,  5.4928,  1.8555,  4.532 ,
  1.4276])
        self.q_table[(2, 7, 0)] = np.array([ 5.2046,  2.5143,  0.814 ,  8.3478,  3.7618,  0.8965, 66.7669,  6.5024,
  5.9509])
        self.q_table[(2, 6, 0)] = np.array([ 1.3021,  4.3062,  3.1116,  1.5643,  0.8643,  8.5747,  7.0322, 52.0994,
  5.3595])
        self.q_table[(3, 1, 0)] = np.array([ 6.9707,  1.8843,  4.0098,  4.5196,  2.0149,  6.1621, 46.1684,  1.676 ,
  3.8315])
        self.q_table[(7, 2, 2)] = np.array([278.9759,   5.0874,   2.4326,   6.156 ,   0.4355,   2.9831,   8.9422,
   1.6598,   3.7769])
        self.q_table[(1, 3, 0)] = np.array([ 3.7568,  4.1337,  5.6298,  5.9646,  8.4894, 14.6681,  4.2348,  0.3616,
  5.0605])

    def discretize_state(self, observation):
        """
        Convert continuous observation to a discrete state tuple using RELATIVE coordinates.
        State: (wind_relative_to_goal, heading_relative_to_goal, speed_bin)
        """
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        # 1. Calculate key vectors/angles
        # Vector/Angle to Goal
        to_goal = self.goal_position - np.array([x, y])
        dist_goal = np.linalg.norm(to_goal)
        if dist_goal < 0.001:
            angle_to_goal = 0.0
        else:
            angle_to_goal = np.arctan2(to_goal[1], to_goal[0])

        # Wind vector angle
        angle_wind = np.arctan2(wy, wx)

        # Velocity vector angle (Heading)
        speed = np.linalg.norm([vx, vy])
        if speed < 0.001:
            angle_heading = angle_to_goal
        else:
            angle_heading = np.arctan2(vy, vx)

        # 2. Compute Relative Angles (normalized to [-pi, pi])

        # Wind Direction relative to Goal Direction
        rel_wind = angle_wind - angle_to_goal
        rel_wind = (rel_wind + np.pi) % (2 * np.pi) - np.pi

        # Heading relative to Goal Direction
        rel_heading = angle_heading - angle_to_goal
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi

        # 3. Discretize
        def get_bin(angle, num_bins):
            # Normalize to [0, 2pi)
            angle = (angle + 2*np.pi) % (2*np.pi)
            # Bin 0 is centered at 0 (from -pi/num_bins to pi/num_bins)
            bin_idx = int((angle + np.pi/num_bins) / (2*np.pi) * num_bins) % num_bins
            return bin_idx

        wind_bin = get_bin(rel_wind, self.angle_bins)
        heading_bin = get_bin(rel_heading, self.angle_bins)

        # Speed bin
        if speed < 0.1:
            speed_bin = 0 # Stopped
        elif speed < 1.0:
            speed_bin = 1 # Slow/Tacking
        else:
            speed_bin = 2 # Fast

        return (wind_bin, heading_bin, speed_bin)

    def act(self, observation, info=None):
        state = self.discretize_state(observation)

        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high

        # Physics filtering
        wx, wy = observation[4], observation[5]
        wind_mag = np.sqrt(wx**2 + wy**2)

        valid_actions = []
        action_vectors = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]

        # Check efficiency for all actions 0-7
        if wind_mag > 0:
            wind_vec = np.array([wx, wy]) / wind_mag
            for i, vec in enumerate(action_vectors):
                vec_np = np.array(vec) / np.linalg.norm(vec)
                eff = calculate_sailing_efficiency(vec_np, wind_vec)

                # LOWERED THRESHOLD to 0.4 to allow 45-degree tacks (eff=0.5)
                if eff >= self.mask_threshold:
                    valid_actions.append(i)

        if not valid_actions: valid_actions = [0,1,2,3,4,5,6,7]

        # Epsilon-Greedy
        if self.np_random.random() < self.exploration_rate:
            action_idx = self.np_random.choice(valid_actions)
        else:
            # Pick best valid action
            q_values = self.q_table[state]
            masked_q = np.full(9, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]
            action_idx = int(np.argmax(masked_q))

        # Calculate VMG for the chosen action for reward shaping
        x, y = observation[0], observation[1]
        position = np.array([x, y])

        # Update VMG tracking for learning
        # prev_vmg holds the VMG of the action from the previous step (s, a)
        # last_vmg holds the VMG of the action from the current step (s', a')
        self.prev_vmg = self.last_vmg

        current_vmg = self._calculate_vmg_reward(position, action_idx, np.array([wx, wy]))
        self.last_vmg = current_vmg

        return int(action_idx)

    def learn(self, state, action, reward, next_state, next_action=None):
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high

        # Reward Shaping: REWARD + VMG_BONUS - PENALTY
        # Use simple shaped reward: Reward + VMG * Multiplier
        # self.prev_vmg corresponds to the action `action` taken at `state`.
        vmg_bonus = self.prev_vmg * 15.0
        step_penalty = 0.5

        shaped_reward = reward + vmg_bonus - step_penalty

        if next_action is not None:
             td_target = shaped_reward + self.discount_factor * self.q_table[next_state][next_action]
        else:
             td_target = shaped_reward + self.discount_factor * np.nanmax(self.q_table[next_state])

        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def reset(self):
        self.last_vmg = 0.0
        self.prev_vmg = 0.0

        # Decay
        self.exploration_rate = max(self.min_exploration,
                                    self.exploration_rate * self.eps_decay_rate)
        self.learning_rate = max(self.min_learning_rate,
                                 self.learning_rate * self.lr_decay_rate)

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate,
                'learning_rate': self.learning_rate
            }, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.exploration_rate = data.get('exploration_rate', 0.01)
            self.learning_rate = data.get('learning_rate', 0.05)

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0) # Action 8: Hold
        ]
        if action < 8:
            return np.array(directions[action], dtype=float)
        return np.array([0, 0], dtype=float)
