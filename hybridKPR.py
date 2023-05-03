import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import os


def exp_cdf(k, dt):
    return 1-np.exp(-dt / k)


class HybridKPR():
    def __init__(self, params: dict = {}):
        self.N = params.get('N', 1)

        self.C = params.get('C', 1)
        self.KON = params.get('KON', 1)
        self.KOFF = params.get('KOFF', 10)
        self.KF = params.get('KF', 1)
        self.BETA = params.get('BETA', 1)
        self.TN = params.get('TN', 1)

        self.STATE = 0
        self.SIGNAL = [0.]
        self.TIME = [0.]
        self.STATE_TRAJ = [0]

        self.DETERMINSTIC_FLAG = True
        self.ALL_DET = False

        self.PHOS_FLAG = False
        self.KP = params.get('KP', 1)

    def reset(self):
        """Sets STATE=0 and clears all records of previous trajectories"""
        self.STATE = 0
        self.SIGNAL = [0.]
        self.TIME = [0.]
        self.STATE_TRAJ = [0]

    def steady_state_occupancies(self):
        x = self.KON * self.C / self.KOFF
        g = self.KOFF / self.KF
        Pss = [1/(1+x)]
        for i in range(1, self.N+1):
            Pss.append(g*x/((1+g)**i * (1+x)))
        Pss.append(x/((1+g)**self.N * (1+x)))
        return Pss

    def get_propensities(self):
        """Computes the probability of each reaction happening"""
        propensities = []
        propensities.append(self.KON * self.C * (self.STATE == 0))  # binding
        for i in range(1, self.N+1):
            propensities.append(self.KOFF * (self.STATE == i))  # unbinding
            propensities.append(self.KF * (self.STATE == i))  # proofread
        # final unbinding
        propensities.append(self.KOFF * (self.STATE == self.N+1))
        if self.PHOS_FLAG:
            propensities.append(self.KP * (self.STATE == self.N+1))  # phos
        return propensities

    def reaction_rules(self, rxn_idx):
        """Returns the new state given the input rxn_idx"""
        # Must match order in get_propensities()
        if rxn_idx == 0:
            return 1
        elif rxn_idx == 2*self.N+1:
            return 0
        elif rxn_idx == 2*self.N+2 and self.PHOS_FLAG:
            return self.STATE
        else:
            if rxn_idx % 2 == 0:
                return self.STATE + 1
            else:
                return 0

    def get_tau(self, prop):
        """"Returns the time til the next reaction, stochastic unless
            the state is determnistic"""
        if self.STATE == self.N + 1 and self.DETERMINSTIC_FLAG:
            # deterministic bound time in final state
            return self.TN
        else:
            # stochastic time to next reaction otherwise
            r1 = np.random.uniform()
            a0 = np.sum(prop)
            return (1./a0)*np.log(1./r1)

    def get_new_state(self, prop):
        """Returns the next state, stochastic unless the state
           is determnistic """
        if self.STATE == self.N + 1 and self.DETERMINSTIC_FLAG:
            # deterministic unbinding
            return 0  # sets self.STATE = 0
        else:
            # stochastic choice of reactions
            r2 = np.random.uniform()
            a0 = np.sum(prop)
            rxn_p = [0] + [p/a0 for p in prop]
            rxn_p = np.cumsum(rxn_p)
            for r in range(len(rxn_p)):
                if rxn_p[r] < r2 and r2 <= rxn_p[r+1]:
                    return self.reaction_rules(r)

    def step(self):
        """Runs the Gillespie algorithm with some determnistic transitions"""
        prop = self.get_propensities()
        # time to next state change
        self.TIME.append(self.TIME[-1] + self.get_tau(prop))
        # update state and signal
        old_state = deepcopy(self.STATE)
        self.STATE = self.get_new_state(prop)
        new_state = self.STATE
        if self.PHOS_FLAG:
            if old_state == self.N + 1 and new_state == self.N + 1:
                self.SIGNAL.append(self.SIGNAL[-1] + 1)
            else:
                self.SIGNAL.append(self.SIGNAL[-1])
        else:
            if old_state == self.N and new_state == self.N + 1:
                self.SIGNAL.append(self.SIGNAL[-1] + self.BETA)
            else:
                self.SIGNAL.append(self.SIGNAL[-1])
        self.STATE_TRAJ.append(self.STATE)

    def sim(self, nsteps, nsims, TQDM=True, ReceptorSS=True):
        """Runs nsteps of state changes, records nsims such stochastic
           trajectories. Returns a record of all trajectories."""
        signal_trajs, time_trajs, state_trajs = [], [], []
        for i in tqdm(range(nsims), disable=(not TQDM)):
            # set initial receptor state according to steady state probs
            if ReceptorSS:
                self.STATE = np.random.choice(self.N+2,
                                             p=self.steady_state_occupancies())
            else:
                self.STATE = 0
            for j in range(nsteps):
                self.step()
            signal_trajs.append(deepcopy(self.SIGNAL))
            time_trajs.append(deepcopy(self.TIME))
            state_trajs.append(deepcopy(self.STATE_TRAJ))
            self.reset()
        return signal_trajs, time_trajs, state_trajs

    def explicit_sim_1(self, t_end, dt, nsims, TQDM=True, ReceptorSS=True):
        """ Simulates the model by taking deterministic time steps dt << 0
        and enacting reactions proportional to dt*k*[c] for rate k and
        concentration [c]. For very small dt this approximates exponential
        waiting time for reactions, which is how the Gillespie algorithm was
        derived anyway."""
        signal_trajs, time_trajs, state_trajs = [], [], []
        for i in tqdm(range(nsims), disable=(not TQDM)):
            # set initial receptor state according to steady state probs
            if ReceptorSS:
                self.STATE = np.random.choice(self.N+2,
                                             p=self.steady_state_occupancies())
            else:
                self.STATE = 0
            # time integration
            while(self.TIME[-1] < t_end):
                # Determine if stochastic dynamics need to be simulated
                skip_flag = False
                if self.DETERMINSTIC_FLAG:
                    if self.STATE == self.N + 1:
                        # deterministic, so next time step and state are known
                        self.TIME.append(self.TIME[-1] + self.TN)
                        # signal was already added
                        self.SIGNAL.append(self.SIGNAL[-1])
                        self.STATE_TRAJ.append(0)
                        self.STATE = 0
                        skip_flag = True
                if not skip_flag:
                    # either determinstic is off, or not in state N+1
                    self.TIME.append(self.TIME[-1] + dt)
                    # get probability of each possible reaction
                    if self.STATE == 0:
                        p = self.KON*self.C*dt
                        if np.random.binomial(1, p) == 0:
                            # nothing happens
                            self.STATE_TRAJ.append(self.STATE)
                            self.SIGNAL.append(self.SIGNAL[-1])
                        else:
                            # ligand binds
                            self.STATE = 1
                            self.STATE_TRAJ.append(self.STATE)
                            if self.N == 0:
                                self.SIGNAL.append(self.SIGNAL[-1] + self.BETA)
                            else:
                                self.SIGNAL.append(self.SIGNAL[-1])
                    elif self.STATE == self.N + 1 and not self.DETERMINSTIC_FLAG:
                        if self.PHOS_FLAG:
                            probs = [1-self.KP*dt-self.KOFF*dt, self.KP*dt, self.KOFF*dt]
                            ch = np.random.choice(3, p=probs)
                            if ch == 0:
                                # nothing happens
                                self.STATE_TRAJ.append(self.STATE)
                                self.SIGNAL.append(self.SIGNAL[-1])
                            elif ch == 1:
                                # phosphorylate a molecule
                                self.STATE_TRAJ.append(self.STATE)
                                self.SIGNAL.append(self.SIGNAL[-1] + 1)
                            else:
                                # ligand unbinds
                                self.STATE = 0
                                self.STATE_TRAJ.append(self.STATE)
                                self.SIGNAL.append(self.SIGNAL[-1])
                        else:
                            p = self.KOFF*dt
                            if np.random.binomial(1, p) == 0:
                                # nothing happens
                                self.STATE_TRAJ.append(self.STATE)
                                self.SIGNAL.append(self.SIGNAL[-1])
                            else:
                                # ligand unbinds
                                self.STATE = 0
                                self.STATE_TRAJ.append(self.STATE)
                                self.SIGNAL.append(self.SIGNAL[-1])
                    else:
                        probs = [1-self.KF*dt-self.KOFF*dt, self.KF*dt, self.KOFF*dt]
                        ch = np.random.choice(3, p=probs)
                        if ch == 0:
                            # nothing happens
                            self.STATE_TRAJ.append(self.STATE)
                            self.SIGNAL.append(self.SIGNAL[-1])
                        elif ch == 1:
                            # next proofreading state
                            self.STATE += 1
                            self.STATE_TRAJ.append(self.STATE)
                            # possibly generate signal
                            if self.STATE == self.N + 1:
                                self.SIGNAL.append(self.SIGNAL[-1] + self.BETA)
                            else:
                                self.SIGNAL.append(self.SIGNAL[-1])
                        else:
                            # ligand unbinds
                            self.STATE = 0
                            self.STATE_TRAJ.append(self.STATE)
                            self.SIGNAL.append(self.SIGNAL[-1])

            time_trajs.append(deepcopy(self.TIME))
            signal_trajs.append(deepcopy(self.SIGNAL))
            state_trajs.append(deepcopy(self.STATE_TRAJ))
            self.reset()
        return signal_trajs, time_trajs, state_trajs

    def explicit_sim_2(self, t_end, _, nsims, TQDM=True, ReceptorSS=True):
        """Simulates the model by drawing times from exponential distribution
        and choosing the reaction according to the shortest time drawn."""
        signal_trajs, time_trajs, state_trajs = [], [], []
        for i in tqdm(range(nsims), disable=(not TQDM)):
            # set initial receptor state according to steady state probs
            if ReceptorSS:
                self.STATE = np.random.choice(self.N+2,
                                             p=self.steady_state_occupancies())
            else:
                self.STATE = 0
            # time integration
            while(self.TIME[-1] < t_end):
                if self.STATE == 0:
                    self.STATE = 1  # receptor will go to state 1
                    tau = np.random.exponential(scale=1.0/(self.KON * self.C))
                    if self.N == 0 and not self.PHOS_FLAG:
                        self.SIGNAL.append(self.SIGNAL[-1] + self.BETA)
                    else:
                        self.SIGNAL.append(self.SIGNAL[-1])
                    self.TIME.append(self.TIME[-1] + tau)
                    self.STATE_TRAJ.append(self.STATE)

                elif self.STATE == self.N + 1:
                    if self.PHOS_FLAG:
                        tau_list = [np.random.exponential(scale=1.0/(self.KP)),
                                    np.random.exponential(scale=1.0/(self.KOFF))]
                        if tau_list[0] > tau_list[1]:
                            # ligand falls off
                            self.STATE = 0
                            self.SIGNAL.append(self.SIGNAL[-1])
                            self.TIME.append(self.TIME[-1] + tau_list[1])
                            self.STATE_TRAJ.append(self.STATE)
                        else:
                            # produce signal
                            self.SIGNAL.append(self.SIGNAL[-1] + 1)
                            self.TIME.append(self.TIME[-1] + tau_list[0])
                            self.STATE_TRAJ.append(self.STATE)

                    else:
                        if self.DETERMINSTIC_FLAG or self.ALL_DET:
                            tau = self.TN
                        else:
                            tau = np.random.exponential(scale=1.0/(self.KOFF))
                        self.STATE = 0  # receptor will go to state 0
                        self.SIGNAL.append(self.SIGNAL[-1])
                        self.TIME.append(self.TIME[-1] + tau)
                        self.STATE_TRAJ.append(self.STATE)
                else:
                    tau_list = [np.random.exponential(scale=1.0/(self.KF)),
                                np.random.exponential(scale=1.0/(self.KOFF))]
                    if tau_list[0] > tau_list[1]:
                        # ligand falls off
                        self.STATE = 0
                        self.SIGNAL.append(self.SIGNAL[-1])
                        if self.ALL_DET:
                            self.TIME.append(self.TIME[-1] + self.TN)
                        else:
                            self.TIME.append(self.TIME[-1] + tau_list[1])
                        self.STATE_TRAJ.append(self.STATE)
                    else:
                        # move to next proofreading state
                        self.STATE += 1
                        if self.STATE == self.N + 1:  # final state
                            self.SIGNAL.append(self.SIGNAL[-1] + self.BETA)
                        else:
                            self.SIGNAL.append(self.SIGNAL[-1])
                        if self.ALL_DET:
                            self.TIME.append(self.TIME[-1] + 1/self.KF)
                        else:
                            self.TIME.append(self.TIME[-1] + tau_list[0])
                        self.STATE_TRAJ.append(self.STATE)
            time_trajs.append(deepcopy(self.TIME))
            signal_trajs.append(deepcopy(self.SIGNAL))
            state_trajs.append(deepcopy(self.STATE_TRAJ))
            self.reset()
        return signal_trajs, time_trajs, state_trajs


def aggregate_trajectories(signal_trajs, time_trajs, state_trajs, tpts=100, target_t=1E16):
    """Align all trajectories to regular time intervals"""
    if target_t != 1E16:
        maxT = target_t
    else:
        maxT = min([max(tj) for tj in time_trajs])
    dT = maxT / tpts
    t_array = np.arange(0., maxT, dT)

    sig_array = []
    state_array = []
    for tj_idx in range(len(time_trajs)):
        signal_aligned = []
        state_aligned = []
        for t_idx in range(1, len(t_array)):
            # get state idx at time t
            for idx, t in enumerate(time_trajs[tj_idx]):
                if t > t_array[t_idx]:
                    if abs(t - t_array[t_idx]) < abs(t - t_array[t_idx-1]):
                        state_idx = idx
                    else:
                        state_idx = idx - 1
                    break
            # record state at time t
            signal_aligned.append(signal_trajs[tj_idx][state_idx])
            state_aligned.append(state_trajs[tj_idx][state_idx])
        # record aligned trajectory into array
        sig_array.append(deepcopy(signal_aligned))
        state_array.append(deepcopy(state_aligned))
    return sig_array, t_array, state_array


def sample_statistics(data):
    """ Returns the mean and variance of the simulation data"""
    return np.mean(data, axis=0), np.var(data, axis=0)


def theory_mean(m, t):
    """ Returns the theory mean based on the input model m and time array t """
    meanX1 = m.KON * m.C * m.KF * m.KOFF * m.BETA * t
    return meanX1 / ((m.KF + m.KOFF) * (m.KOFF + m.C * m.KON))


def theory_mean_n(m, t):
    """ Returns the theory mean based on the input model m and time array t """
    meanX1 = m.KON * m.C * m.KP * m.BETA / (m.KOFF + m.C * m.KON)
    f = (1. / (1. + m.KOFF / m.KF))**m.N
    return meanX1 * f * t


def test_timecourse(DET_FLAG=True, PHOS_FLAG=False):
    KPR1 = HybridKPR()
    if not DET_FLAG:
        KPR1.DETERMINSTIC_FLAG = False
    if PHOS_FLAG:
        KPR1.PHOS_FLAG = True
    signal_trajs, time_trajs, state_trajs = KPR1.sim(100, 500)
    sig_array, t_array, state_array = aggregate_trajectories(signal_trajs,
                                                             time_trajs,
                                                             state_trajs)
    mean_sig, var_sig = sample_statistics(sig_array)
    plt.plot(t_array[:-1], mean_sig, 'b', label='simulation')
    plt.fill_between(t_array[:-1], mean_sig+np.sqrt(var_sig),
                     mean_sig-np.sqrt(var_sig), color='b', alpha=0.25)
    if KPR1.PHOS_FLAG:
        plt.plot(t_array[:-1], theory_mean_n(KPR1, t_array[:-1]),
                 'k--', label='theory')
    else:
        plt.plot(t_array[:-1], theory_mean(KPR1, t_array[:-1]),
                 'k--', label='theory')
    plt.legend()
    plt.show()


def test_doseresponse(nsteps, nsims):
    crange = np.logspace(-2, 2, 50)
    minT = 1E16  # the latest possible time to plot in all simulations
    dose_response = []
    response_var = []
    time_array = []
    for c in tqdm(range(len(crange))):
        KPR1 = HybridKPR()
        KPR1.C = crange[c]
        signal_trajs, time_trajs, state_trajs = KPR1.sim(nsteps, nsims, TQDM=False)
        sig_array, t_array, state_array = aggregate_trajectories(signal_trajs,
                                                                 time_trajs,
                                                                 state_trajs)
        mean_sig, var_sig = sample_statistics(sig_array)
        dose_response.append(mean_sig)
        response_var.append(var_sig)
        time_array.append(t_array)

        # adjust time of dose-response accordingly
        if max(t_array) < minT:
            minT = deepcopy(max(t_array))
    # get response at minT
    mean = []
    var = []
    # t = [i for i in time_array[0] if i <= minT]
    for idx, r in enumerate(dose_response):
        idx2 = min(range(len(time_array)), key=lambda i: abs(time_array[idx][i]-minT))
        mean.append(dose_response[idx][idx2])
        var.append(response_var[idx][idx2])

    fig, ax = plt.subplots()
    plt.plot(crange, mean, 'b', label='simulation')
    ax.set_xscale('log')
    plt.fill_between(crange, mean+np.sqrt(var),
                     mean-np.sqrt(var), color='b', alpha=0.25)
    plt.show()


def HN_vs_N(nsteps, nsims, DETERMINSTIC_FLAG=True, PHOS_FLAG=False):
    Nrange = list(range(5))

    def __HN_vs_N_mean_var(KPR1, KPR2):
        minT = 1E16
        dose_response_1, dose_response_2 = [], []
        response_var_1, response_var_2 = [], []
        time_array_1, time_array_2 = [], []

        for N in tqdm(Nrange):
            for idx, m in enumerate([KPR1, KPR2]):
                m.N = N
                sts, tts, stts = m.sim(nsteps, nsims, TQDM=False)
                sig, t_arr, _ = aggregate_trajectories(sts, tts, stts)
                mean_sig, var_sig = sample_statistics(sig)
                if idx == 0:
                    dose_response_1.append(mean_sig)
                    response_var_1.append(var_sig)
                    time_array_1.append(t_arr)
                else:
                    dose_response_2.append(mean_sig)
                    response_var_2.append(var_sig)
                    time_array_2.append(t_arr)
                # adjust time of dose-response accordingly
                if max(t_arr) < minT:
                    minT = deepcopy(max(t_arr))
        # get response at minT
        mean_1, mean_2 = [], []
        var_1, var_2 = [], []
        # t = [i for i in time_array[0] if i <= minT]
        for idx, r in enumerate(dose_response_1):
            idx2 = min(range(len(time_array_1)), key=lambda i: abs(time_array_1[idx][i]-minT))
            mean_1.append(dose_response_1[idx][idx2])
            var_1.append(response_var_1[idx][idx2])
        for idx, r in enumerate(dose_response_2):
            idx2 = min(range(len(time_array_2)), key=lambda i: abs(time_array_2[idx][i]-minT))
            mean_2.append(dose_response_2[idx][idx2])
            var_2.append(response_var_2[idx][idx2])

        return np.array(mean_1), np.array(var_1), np.array(mean_2), np.array(var_2)

    # Compute FLD
    KPRk1 = HybridKPR({'KOFF': 10.})
    KPRk2 = HybridKPR({'KOFF': 1.})
    if not DETERMINSTIC_FLAG:
        KPRk1.DETERMINSTIC_FLAG = False
        KPRk2.DETERMINSTIC_FLAG = False
    if PHOS_FLAG:
        KPRk1.PHOS_FLAG = True
        KPRk2.PHOS_FLAG = True

    # Compute Hopfield Ninio first
    g = deepcopy(KPRk1.KOFF / KPRk1.KF)
    HN = [((1+g)/(1+0.1*g))**i for i in Nrange]

    m1, v1, m2, v2 = __HN_vs_N_mean_var(KPRk1, KPRk2)
    HN_sim = m2 / m1
    HN_ratio = [f / HN_sim[0] for f in HN_sim]

    # print results to copy over to Mathematica
    txt = "{"
    for i in range(len(HN_ratio)):
        txt = txt + "{{{0}, {1:.2f}}},".format(i, HN_ratio[i])
    txt = txt[:-1] + "}"
    print(txt)

    fig, ax = plt.subplots()
    plt.plot(Nrange, HN_ratio, 'b', label='FLD')
    plt.plot(Nrange, HN, 'k--', label='HN')
    plt.legend()
    ax.set_yscale('log')
    plt.show()


def FLD_vs_N(nsims, DETERMINSTIC_FLAG=True, target_time=100., ReceptorSS=True, plotFLAG=True, PHOS_FLAG=False, params={'KOFF': 10.}, f=0, dt=0.0001, NTEST=5, zeta=0.1, fname='', TAU_KOFF=False, ALL_DET=False, TQDM=False):
    Nrange = list(range(NTEST))
    FLD, HN = [], []
    # Compute FLD
    for n in tqdm(Nrange):
        params.update({'N': n})
        KPRk1 = HybridKPR(params)
        KPRk2 = HybridKPR(params)
        KPRk2.KOFF = KPRk1.KOFF * zeta
        if not DETERMINSTIC_FLAG:
            KPRk1.DETERMINSTIC_FLAG = False
            KPRk2.DETERMINSTIC_FLAG = False
        if PHOS_FLAG:
            KPRk1.PHOS_FLAG = True
            KPRk2.PHOS_FLAG = True
        if TAU_KOFF:
            KPRk1.TN = 1./KPRk1.KOFF
            KPRk2.TN = 1./KPRk2.KOFF
        if ALL_DET:
            KPRk1.ALL_DET = True
            KPRk2.ALL_DET = True

        # Run explicit simulations
        if f == 0 and not ALL_DET:
            nsteps = int(target_time*KPRk2.KOFF*20)
            signal_trajs_1, time_trajs_1, state_trajs_1 = KPRk1.sim(nsteps, nsims, TQDM=TQDM, ReceptorSS=ReceptorSS)
            signal_trajs_2, time_trajs_2, state_trajs_2 = KPRk2.sim(nsteps, nsims, TQDM=TQDM, ReceptorSS=ReceptorSS)
            # quality control to make sure we ran enough steps in Gillespie
            tend_avg_1 = np.mean(np.array(time_trajs_1)[:,-1])
            tend_avg_2 = np.mean(np.array(time_trajs_2)[:,-1])
            if tend_avg_1 < target_time or tend_avg_2 < target_time:
                print(np.mean(np.array(time_trajs_1)[:,-1]))
                print(np.mean(np.array(time_trajs_2)[:,-1]))
        else:
            if f == 1 and not ALL_DET:
                def __func__(m):
                    return m.explicit_sim_1
            elif f == 2 or ALL_DET:
                def __func__(m):
                    return m.explicit_sim_2

            signal_trajs_1, time_trajs_1, state_trajs_1 = __func__(KPRk1)(target_time, dt, nsims, TQDM=TQDM, ReceptorSS=ReceptorSS)
            signal_trajs_2, time_trajs_2, state_trajs_2 = __func__(KPRk2)(target_time, dt, nsims, TQDM=TQDM, ReceptorSS=ReceptorSS)

        sig_array_1, t_array_1, state_array_1 = aggregate_trajectories(signal_trajs_1, time_trajs_1, state_trajs_1, target_t=target_time)
        mean_sig_1, var_sig_1 = sample_statistics(sig_array_1)

        sig_array_2, t_array_2, state_array_2 = aggregate_trajectories(signal_trajs_2, time_trajs_2, state_trajs_2, target_t=target_time)
        mean_sig_2, var_sig_2 = sample_statistics(sig_array_2)

        m1 = mean_sig_1[-1]
        v1 = var_sig_1[-1]
        m2 = mean_sig_2[-1]
        v2 = var_sig_2[-1]
        print(m1, m2)

        FLD.append(np.square(np.subtract(m1, m2)) / (v1 + v2))
        HN.append(m2/m1)

    FLD_ratio = [f / FLD[0] for f in FLD]
    HN_ratio = [f / HN[0] for f in HN]

    # print results to copy over to Mathematica
    txt = "{"
    for i in range(len(FLD_ratio)):
        txt = txt + "{{{0}, {1:.4f}}},".format(i, FLD_ratio[i])
    txt = txt[:-1] + "}"
    print(txt)

    # print HN results to copy over to Mathematica
    txt2 = "HN: {"
    for i in range(len(HN_ratio)):
        txt2 = txt2 + "{{{0}, {1:.2f}}},".format(i, HN_ratio[i])
    txt2 = txt2[:-1] + "}"
    print(txt2)

    # save results
    txt += "\n\n{"
    txt += 'C: ' + str(KPRk1.C) + ', KON: ' + str(KPRk1.KON)
    txt += ', KOFF:' + str(KPRk1.KOFF) + ', ZETA: ' + str(zeta)
    txt += ', KF: ' + str(KPRk1.KF) + ', BETA: ' + str(KPRk1.BETA)
    txt += ', TN: ' + str(KPRk1.TN)
    txt += ', DETERMINSTIC_FLAG: ' + str(KPRk1.DETERMINSTIC_FLAG)
    txt += ', PHOS_FLAG: ' + str(KPRk1.PHOS_FLAG)
    txt += ', KP: ' + str(KPRk1.KP)
    txt += '}\n'

    if fname == '':
        fname = os.path.join(os.getcwd(), 'output', 'FLD_vs_N_output.txt')
    else:
        fname = os.path.join(os.getcwd(), 'output', fname)
    with open(fname, "w") as text_file:
        text_file.write(txt)

    if plotFLAG:
        fig, ax = plt.subplots()
        plt.plot(Nrange, FLD_ratio, 'b', label='FLD')
        plt.plot(Nrange, HN_ratio, 'k--', label='HN')
        plt.legend()
        ax.set_yscale('log')
        plt.show()


def compare_sims(t_end, dt, nsims, DETERMINSTIC_FLAG=True, PHOS_FLAG=False, fname=''):
    # Model initialization
    KPR1 = HybridKPR()
    KPR2 = HybridKPR()
    KPR3 = HybridKPR()
    if not DETERMINSTIC_FLAG:
        KPR1.DETERMINSTIC_FLAG = False
        KPR2.DETERMINSTIC_FLAG = False
        KPR3.DETERMINSTIC_FLAG = False
    if PHOS_FLAG:
        KPR1.PHOS_FLAG = True
        KPR2.PHOS_FLAG = True
        KPR3.PHOS_FLAG = True

    # Run explicit simulations
    signal_trajs_1, time_trajs_1, state_trajs_1 = KPR1.explicit_sim_1(t_end, dt, nsims)
    signal_trajs_2, time_trajs_2, state_trajs_2 = KPR2.explicit_sim_2(t_end, None, nsims)

    sig_array_1, t_array_1, state_array_1 = aggregate_trajectories(signal_trajs_1, time_trajs_1, state_trajs_1, target_t=t_end)
    mean_sig_1, var_sig_1 = sample_statistics(sig_array_1)

    sig_array_2, t_array_2, state_array_2 = aggregate_trajectories(signal_trajs_2, time_trajs_2, state_trajs_2, target_t=t_end)
    mean_sig_2, var_sig_2 = sample_statistics(sig_array_2)

    # Run Gillespie simulation
    signal_trajs_3, time_trajs_3, state_trajs_3 = KPR3.sim(int(t_end/dt), nsims)
    sig_array_3, t_array_3, state_array_3 = aggregate_trajectories(signal_trajs_3, time_trajs_3, state_trajs_3, target_t=t_end)
    mean_sig_3, var_sig_3 = sample_statistics(sig_array_3)

    # Plot to compare
    fig, ax = plt.subplots()
    # if PHOS_FLAG:
    #     plt.plot(t_array_1[:-1], theory_mean_n(KPR1, t_array_1[:-1]), 'k--', label='Theory')
    # else:
    #     plt.plot(t_array_1[:-1], theory_mean(KPR1, t_array_1[:-1]), 'k--', label='Theory')
    plt.plot(t_array_1[:-1], mean_sig_1, '#83b6d4', label='Deterministic time steps')
    plt.fill_between(t_array_1[:-1], mean_sig_1+np.sqrt(var_sig_1),
                     mean_sig_1-np.sqrt(var_sig_1), color='#83b6d4', alpha=0.25)
    plt.plot(t_array_2[:-1], mean_sig_2, '#e57290', label='Exponential waiting')
    plt.fill_between(t_array_2[:-1], mean_sig_2+np.sqrt(var_sig_2),
                     mean_sig_2-np.sqrt(var_sig_2), color='#e57290', alpha=0.25)
    plt.plot(t_array_3[:-1], mean_sig_3, '#9cb9a7', label='Gillespie')
    plt.fill_between(t_array_3[:-1], mean_sig_3+np.sqrt(var_sig_3),
                     mean_sig_3-np.sqrt(var_sig_3), color='#9cb9a7', alpha=0.25)

    plt.legend()
    plt.ylim(0.)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if fname == '':
        fname = os.path.join(os.getcwd(), 'output', 'hybrid_sim_validation.pdf')
    else:
        fname = os.path.join(os.getcwd(), 'output', fname)
    plt.savefig(fname)


if __name__ == '__main__':
    # Compare simulation algorithms for hybrid model
    # compare_sims(50, 0.0001, 100, DETERMINSTIC_FLAG=True)
    # Compare simulation algorithms for full stochastic + phosphorylation model
    # compare_sims(0.5, 0.0001, 500, DETERMINSTIC_FLAG=False, PHOS_FLAG=True, fname='stoch_sim_phos_validation.pdf')

    # Short time FLD of the fully stochastic model
    # FLD_vs_N(5000, DETERMINSTIC_FLAG=False, target_time=0.5, ReceptorSS=False, plotFLAG=False, PHOS_FLAG=True, f=2, dt=0.0001)
    # FLD_vs_N(5000, DETERMINSTIC_FLAG=False, target_time=5, ReceptorSS=False, plotFLAG=False, PHOS_FLAG=True, f=2, dt=0.0001)
    # FLD_vs_N(5000, DETERMINSTIC_FLAG=False, target_time=50, ReceptorSS=False, plotFLAG=False, PHOS_FLAG=True, f=2, dt=0.0001)

    # FLD with tau = 1/koff
    # FLD_vs_N(100, DETERMINSTIC_FLAG=True, target_time=100,
    #          ReceptorSS=True, plotFLAG=False, params={'BETA': 10, 'C': 0.5, 'KF': 1., 'KOFF': 10},
    #          f=0, fname='FLD_vs_N_t_is_1_over_koff.txt', dt=0.001,
    #          TAU_KOFF=True, NTEST=6)

    # FLD with tau = 1/koff and at low g
    # FLD_vs_N(100, DETERMINSTIC_FLAG=True, target_time=100,
    #          ReceptorSS=True, plotFLAG=False, params={'BETA': 10, 'C': 0.5, 'KF': 10./3., 'KOFF': 10.},
    #          f=0, fname='FLD_vs_N_t_is_1_over_koff_low_g.txt', dt=0.001,
    #          TAU_KOFF=True, NTEST=2)

    # g=10 and high concentration
    # FLD_vs_N(100, DETERMINSTIC_FLAG=True, target_time=100,
    #          ReceptorSS=True, plotFLAG=False, params={'BETA': 10, 'C': 5, 'KF': 1., 'KOFF': 10},
    #          f=0, fname='FLD_vs_N_t_is_1_over_koff.txt', dt=0.001,
    #          TAU_KOFF=True, NTEST=6)
    #
    # # g=10 and low concentration
    # FLD_vs_N(100, DETERMINSTIC_FLAG=True, target_time=100,
    #          ReceptorSS=True, plotFLAG=False, params={'BETA': 10, 'C': 0.1, 'KF': 1., 'KOFF': 10},
    #          f=0, fname='FLD_vs_N_t_is_1_over_koff.txt', dt=0.001,
    #          TAU_KOFF=True, NTEST=6)

    # FLD with all steps except binding deterministic
    # FLD_vs_N(200, DETERMINSTIC_FLAG=True, target_time=100,
    #          ReceptorSS=True, plotFLAG=False,
    #          params={'BETA': 10, 'C': 0.5, 'KF': 10./3., 'KOFF': 10.},
    #          f=2, fname='FLD_vs_N_all_deterministic_g3.txt', dt=0.001,
    #          ALL_DET=True, TAU_KOFF=True)

    # FLD_vs_N(100, DETERMINSTIC_FLAG=True, target_time=100,
    #          params={'BETA': 10, 'C': 0.5, 'KF': 1., 'KOFF': 10},
    #          f=2, fname='FLD_vs_N_all_deterministic_g10.txt', dt=0.001,
    #          ALL_DET=True, TAU_KOFF=True)
