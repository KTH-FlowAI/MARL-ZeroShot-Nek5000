class colorplate:
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian
    gray    = "#DFE0D9" # ermuyu 
    grays    = "#6B6C6E" # ermuyu 
    green   = "#16A951"     # Shi Lv
    deepgreen   = "#057748" # Song Hua Lv 
    lightblue = '#3EEDE7' # Bi Lan
    pink = '#FF0097' # Yanghong 
    darkred = '#9D2933' # Yanzhi 
    deepblue = '#003371' # Qianqing
    brown  = "#9F6027"
    purple = "#A76283" # zi jing pin feng 
    orange = "#EA5514" # huang dan
    deeppurple = "#674196"
    yellow2 = '#EBD842'

def plt_setUp():
    import matplotlib.pyplot as plt
    plt.rc("font",family = "serif")
    #plt.rc("text",usetex = "true")
    plt.rc("font",size = 20)
    plt.rc("axes",labelsize = 30, linewidth = 2)
    plt.rc("legend",fontsize= 20, handletextpad = 0.1)
    plt.rc("xtick",labelsize = 25)
    plt.rc("ytick",labelsize = 25)
    return 



def make_style(color, alpha, marker, label, lw=2.0, linestyle='-', markersize=7.0, markevery=20):
    return {
        'lw': lw,
        'c': color,
        'linestyle': linestyle,
        'alpha': alpha,
        'marker': marker,
        'markersize': markersize,
        'markevery': markevery,
        'label': label,
    }
