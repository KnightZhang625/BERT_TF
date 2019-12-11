import matplotlib.pyplot as plt
import codecs as co
import os
import sys

if len(sys.argv) > 1:
    LF = sys.argv[1]
else: 
    LF = './loss_record'
AF = './infer_precision'
# ADF = './ad_losses'
# GNG = './gng_losses'

plt.figure(figsize=(22,12))

def full_arr(arr, target):
    while(len(arr) < target):
        arr.append(arr[-1])
    return arr

def draw_new_loss():
    global plt
    if (not os.path.exists(LF)):
        print(LF + ' does not exists.')
        return
    lines = co.open(LF).readlines()
    _pre_losses = []
    _dis_losses = []
    _sup_losses = []
    _adv_osses = []
    _lanlosses = []
    c = 0
    for line in lines:
        line = line.replace('\n', '')
        line = line.split(',')
        for l in line:
            l = l.strip()
            l = l.split(':')
            if (l[0].strip() == 'pre_avg'):
                _pre_losses.append(float(l[1].strip()))
            elif (l[0].strip() == 'dis_avg'):
                _dis_losses.append(float(l[1].strip()))
            elif (l[0].strip() == 'sup_avg'):
                if c == 1:
                    _sup_losses.append(float(l[1].strip()))
                    c = 0
                else:
                    c +=1
            elif (l[0].strip() == 'adv_avg'):
                _adv_osses.append(float(l[1].strip()))
        ma = max([len(_pre_losses), len(_dis_losses), len(_sup_losses), len(_adv_osses), len(_lanlosses)])
        if (len(_pre_losses) > 0):
            _pre_losses = full_arr(_pre_losses, ma)
        if (len(_dis_losses) > 0):
            _dis_losses = full_arr(_dis_losses, ma)
        if (len(_sup_losses) > 0):
            _sup_losses = full_arr(_sup_losses, ma)
        if (len(_adv_osses) > 0):
            _adv_osses = full_arr(_adv_osses, ma)
        if (len(_lanlosses) > 0):
            _lanlosses = full_arr(_lanlosses, ma)

    plot_id = 220
    # if (len(_pre_losses) > 0):
    #     plot_id += 1
    #     pre_sp = plt.subplot(231)
    #     pre_sp.set_title('pre loss')
    #     plt.plot(range(len(_pre_losses)), _pre_losses, 'k-', lw=2)
    # if (len(_dis_losses) > 0):
    #     plot_id += 1
        # cls_sp = plt.subplot(231)
        # cls_sp.set_title('ad loss')
        # plt.plot(range(len(_dis_losses)), _dis_losses, 'g-', lw=2)
    if (len(_sup_losses) > 0):
        plot_id += 1
    #     ad_sp = plt.subplot(232)
    #     ad_sp.set_title('final loss')
        plt.plot(range(len(_sup_losses)), _sup_losses, 'r-', lw=1)
    # if (len(_adv_osses) > 0):
    #     plot_id += 1
    #     c_sp = plt.subplot(234)
    #     c_sp.set_title('c loss')
        # plt.plot(range(len(_adv_osses)), _adv_osses, 'r-', lw=2)
    # if (len(_lanlosses) > 0):
    #     plot_id += 1
    #     lan_sp = plt.subplot(235)
    #     lan_sp.set_title('lan loss')
    #     plt.plot(range(len(_lanlosses)), _lanlosses, 'k-', lw=2)

# def draw_loss():
#     global plt
#     if (not os.path.exists(LF)):
#         print(LF + ' does not exists.')
#         return
#     lines = co.open(LF).readlines()
#     data1 = []
#     data2 = []
#     for i in range(len(lines)):
#         line = lines[i].strip()
#         line = line.split(' : ')[1]
#         line = line.split(',')
#         data1.append(float(line[0]))
#         data2.append(float(line[1]))
#     assert len(data1) == len(data2)
#     x = range(1, len(data1) + 1)
#     sp = plt.subplot(221)
#     sp.set_title('main loss and z loss')
#     plt.plot(x, data1, 'r-', lw=2)
#     plt.plot(x, data2, 'b-', lw=2)
#     print('MIN LOSS: ' + str(min(data1)))
#     print('MIN LANTENT: ' + str(min(data2)))
#     # plt.show()

def draw_acc():
    global plt
    if (not os.path.exists(AF)):
        print(AF + ' does not exists.')
        return
    lines = co.open(AF).readlines()
    data1 = []
    for i in range(len(lines)):
        line = lines[i].strip()
        line = line.split(' : ')[1]
        # line = line.split(',')
        try:
            data1.append(float(line))
        except:
            continue
    x = range(1, len(data1) + 1)
    sp = plt.subplot(236)
    sp.set_title('accuracy, 1.0 is not the best.')
    plt.plot(x, data1, 'g-', lw=2)
    print('MAX PRECISION: ' + str(max(data1)))
    # plt.show()

# def draw_adlosses():
#     global plt
#     if (not os.path.exists(ADF)):
#         print(ADF + ' does not exists.')
#         return
#     lines = co.open(ADF).readlines()
#     data1 = []
#     for i in range(len(lines)):
#         line = lines[i].strip()
#         line = line.replace('\n', '')
#         # line = line.split(',')
#         try:
#             data1.append(float(line))
#         except:
#             continue
#     x = range(1, len(data1) + 1)
#     sp = plt.subplot(223)
#     sp.set_title('advesarial loss')
#     plt.plot(x, data1, 'c-', lw=2)
#     print('MIN AD_LOSS: ' + str(min(data1)))

# def draw_gng():
#     global plt
#     if (not os.path.exists(GNG)):
#         print(GNG + ' does not exists.')
#         return
#     lines = co.open(GNG).readlines()
#     data1 = []
#     data2 = []
#     for i in range(len(lines)):
#         line = lines[i].strip()
#         line = line.replace('\n', '')
#         line = line.split(',')
#         try:
#             data1.append(float(line[0]))
#             data2.append(float(line[1]))
#         except:
#             continue
#     x = range(1, len(data1) + 1)
#     sp = plt.subplot(224)
#     sp.set_title('discriminator loss')
#     plt.plot(x, data1, 'm-', lw=2)
#     plt.plot(x, data2, 'k-', lw=2)

# draw_loss()
# draw_acc()
# draw_adlosses()
# draw_gng()
draw_new_loss()
# draw_acc()
plt.grid(True)
plt.show()
