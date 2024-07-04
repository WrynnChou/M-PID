import numpy as np
import matplotlib.pyplot as plt



# ------------------------------------------ SGD Optimization --------------------------------------------

def linear_sgd(step=0.01):
    x_0 = np.array([[0.], [0.]])
    err = []
    x = []
    y_sys = []
    i = 0
    for i in range(100):
        y = sys(x_0)
        x_inner = x_0
        y_true = y_ref[:, i].reshape([2, 1])
        for j in range(T):
            y = sys(x_inner)
            delta = 2 * np.matmul(A, y_true - y)
            x_inner += delta * step
            err_ab = np.sum(np.abs(sys(x_inner) - y_true))
            if err_ab < 0.01:
                break

        x_0 = x_inner
        x.append(x_0.copy())
        err.append(y_ref[:, i].reshape([2, 1]) - y)
        y_sys.append(y)
    x = np.array(x)
    y_sys = np.array(y_sys)
    err = np.array(err)
    x = x.reshape((100, 2))
    y_sys = y_sys.reshape((100, 2))
    a_e = np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x)), np.matmul(np.transpose(x), y_sys))

    plt.figure()
    plt.plot(t[1:], y_sys[:, 0], c='blue')
    plt.plot(t, y_ref[0, :], c='r')
    plt.plot(t[1:], y_sys[:, 1], c="blue", linestyle='--')
    plt.plot(t, y_ref[1, :], c='r', linestyle='--')
    plt.title("Linear + SGD")
    plt.show(block=True)


# ---------------------- PID ------------------------

class PID_Prama:
    def __init__(self):
        self.Kp = 0
        self.Ki = 0
        self.Kd = 0
        self.set_val = 0
        self.error_last = 0
        self.error_prev = 0
        self.error_sum = 0


def PID_Controller_Increa(pid: PID_Prama, out_now: float, target: float) -> float:
    """
    增量计算公式：Pout=Kp*[e(t) - e(t-1)] + Ki*e(t) + Kd*[e(t) - 2*e(t-1) +e(t-2)]
    pid：pid算法的参数,
    output_now: 当前系统的输出
    index: 当前系统输出对应的输入的分片索引
    参考 https://c.miaowlabs.com/E08.html
    https://www.cnblogs.com/uestc-mm/p/10512333.html
    """
    error = target - out_now
    Res = pid.Kp * (error - pid.error_last) + pid.Ki * error + pid.Kd * (error - 2 * pid.error_last + pid.error_prev)
    # Res = (Kp + Ki + Kd) * error + (-Kp - 2*Kd) * error_last + Kd * error_prev
    pid.error_prev = pid.error_last
    pid.error_last = error
    return Res


def MPC(x1, x2, alpha=0.9, beta=0.9):
    return np.array([alpha * x1 + (1 - alpha) * x2, beta * x2 + (1 - beta) * x1])


def i_pid(kp=0.01, ki=0.01, kd=0.01, kp2=0.01, ki2=0.01, kd2=0.01):
    PID_val1 = PID_Prama()
    PID_val2 = PID_Prama()
    # PID参数
    PID_val1.Kp = kp
    PID_val1.Ki = ki
    PID_val1.Kd = kd

    PID_val2.Kp = kp2
    PID_val2.Ki = ki2
    PID_val2.Kd = kd2
    x_0 = np.array([[0.], [0.]])

    sys_in = [x_0]
    sys_out = []
    sys_err = []
    for i in range(100):
        target1, target2 = y_ref[0, i], y_ref[1, i]
        out = sys(x_0)
        diff1 = PID_Controller_Increa(PID_val1, out[0], target1)
        diff2 = PID_Controller_Increa(PID_val2, out[1], target2)
        x_0 += np.array([diff1, diff2])
        sys_in.append(x_0)
        sys_out.append(out)
        sys_err.append(y_ref[:, i].reshape([2, 1]) - out)

    sys_in = np.array(sys_in)
    sys_out = np.array(sys_out)
    sys_err = np.array(sys_err)

    plt.figure()
    plt.plot(t[1:], sys_out[:, 0], c='blue')
    plt.plot(t, y_ref[0, :], c='r')
    plt.plot(t[1:], sys_out[:, 1], c='blue', linestyle="--")
    plt.plot(t, y_ref[1, :], c='r', linestyle="--")
    plt.title("I-PID")
    plt.show(block=True)


def m_pid(kp=0.01, ki=0.01, kd=0.01, kp2=0.01, ki2=0.01, kd2=0.01, alpha=0.9, beta=0.9):
    PID_val1 = PID_Prama()
    PID_val2 = PID_Prama()
    # PID参数
    PID_val1.Kp = kp
    PID_val1.Ki = ki
    PID_val1.Kd = kd

    PID_val2.Kp = kp2
    PID_val2.Ki = ki2
    PID_val2.Kd = kd2
    x_0 = np.array([[0.], [0.]])

    sys_in = [x_0]
    sys_out = []
    sys_err = []
    for i in range(100):
        target1, target2 = y_ref[0, i], y_ref[1, i]
        out = sys(x_0)

        diff1 = PID_Controller_Increa(PID_val1, out[0], target1)
        diff2 = PID_Controller_Increa(PID_val2, out[1], target2)
        x_0 += MPC(diff1, diff2, alpha, beta)
        sys_in.append(x_0)
        sys_out.append(out)
        sys_err.append(y_ref[:, i].reshape([2, 1]) - out)

    sys_in = np.array(sys_in)
    sys_out = np.array(sys_out)
    sys_err = np.array(sys_err)

    plt.figure()
    plt.plot(t[1:], sys_out[:, 0], c='blue')
    plt.plot(t, y_ref[0, :], c='r')
    plt.plot(t[1:], sys_out[:, 1], c='blue', linestyle="--")
    plt.plot(t, y_ref[1, :], c='r', linestyle="--")
    plt.title("M-PID")
    plt.show(block=True)


def m_pid2(kp=0.01, ki=0.01, kd=0.01, kp2=0.01, ki2=0.01, kd2=0.01, alpha=[0.9, 0.9], beta=[0.9, 0.9]):
    PID_val1 = PID_Prama()
    PID_val2 = PID_Prama()
    # PID参数
    PID_val1.Kp = kp
    PID_val1.Ki = ki
    PID_val1.Kd = kd

    PID_val2.Kp = kp2
    PID_val2.Ki = ki2
    PID_val2.Kd = kd2
    x_0 = np.array([[0.], [0.]])

    sys_in = [x_0]
    sys_out = []
    sys_err = []
    for i in range(100):
        target1, target2 = y_ref[0, i], y_ref[1, i]
        out = sys(x_0)

        diff1 = PID_Controller_Increa(PID_val1, MPC(out[0], out[1], alpha[0], beta[0])[0],
                                      MPC(target1, target2, alpha[0], beta[0])[0])
        diff2 = PID_Controller_Increa(PID_val2, MPC(out[0], out[1], alpha[0], beta[0])[1],
                                      MPC(target1, target2, alpha[0], beta[0])[1])
        x_0 += MPC(diff1, diff2, alpha[1], beta[1])
        sys_in.append(x_0)
        sys_out.append(out)
        sys_err.append(y_ref[:, i].reshape([2, 1]) - out)

    sys_in = np.array(sys_in)
    sys_out = np.array(sys_out)
    sys_err = np.array(sys_err)

    plt.figure()
    plt.plot(t[1:], sys_out[:, 0], c='blue')
    plt.plot(t, y_ref[0, :], c='r')
    plt.plot(t[1:], sys_out[:, 1], c='blue', linestyle="--")
    plt.plot(t, y_ref[1, :], c='r', linestyle="--")
    plt.title("M-PID")
    plt.show(block=True)


if __name__ == '__main__':

    def sys(x):
        return np.matmul(A, x) + b

    A = np.array([[1, 0.2], [0.2, 2]])
    b = np.array([[2], [3]])

    T = 10  # 100
    t = np.linspace(0, 10, 101)
    # y_ref = np.array([[10],[5]]) * np.ones([2, 101])
    y_ref = np.array([[t * 7 + 10 * np.sin(t)], [t * 3 + 5 * np.cos(t)]])
    y_ref = y_ref.reshape([2, 101])

    linear_sgd()
    i_pid(0.35, 0.25, 0.15, 0.1, 0.1, 0.1)
    m_pid(0.35, 0.25, 0.15, 0.1, 0.1, 0.1)










    print('Good luck!')
