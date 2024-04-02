import imp_matrix
import pyomo.environ as pyo 
import math
import numpy as np
import pandas as pd
from timeit import default_timer as timer

def opf_3ph_lindistflow(network, load_curve_a, load_curve_b, load_curve_c, der_a, der_b, der_c, vm_pu):
    
    def voltage_magnitude_lb(m, n, p, t):
        return 0.81 <= m.voltage[n,p,t]

    def voltage_magnitude_ub(m, n, p,t):
        return m.voltage[n,p,t] <= 1.21
    
    def voltage_drop_ft(m, f_n, t_n, p, t):
        
        if p == 0:
        
            return m.voltage[t_n,p,t] - m.voltage[f_n,p,t] \
                - (-2*r_abc[f_n,t_n][0,0]*m.p_line[f_n,t_n,0,t] \
                   + (r_abc[f_n,t_n][0,1] - math.sqrt(3)*x_abc[f_n,t_n][0,1])*m.p_line[f_n,t_n,1,t] \
                   + (r_abc[f_n,t_n][0,2] + math.sqrt(3)*x_abc[f_n,t_n][0,2])*m.p_line[f_n,t_n,2,t])\
                 -(-2*x_abc[f_n,t_n][0,0]*m.q_line[f_n,t_n,0,t] \
                   + (x_abc[f_n,t_n][0,1] + math.sqrt(3)*r_abc[f_n,t_n][0,1])*m.q_line[f_n,t_n,1,t] \
                   + (x_abc[f_n,t_n][0,2] - math.sqrt(3)*r_abc[f_n,t_n][0,2])*m.q_line[f_n,t_n,2,t]) == 0
        
        elif p == 1:
        
            return m.voltage[t_n,p,t] - m.voltage[f_n,p,t] \
                -(-2*r_abc[f_n,t_n][1,1]*m.p_line[f_n,t_n,1,t] \
                   + (r_abc[f_n,t_n][1,0] + math.sqrt(3)*x_abc[f_n,t_n][1,0])*m.p_line[f_n,t_n,0,t] \
                   + (r_abc[f_n,t_n][1,2] - math.sqrt(3)*x_abc[f_n,t_n][1,2])*m.p_line[f_n,t_n,2,t])\
                 -(-2*x_abc[f_n,t_n][1,1]*m.q_line[f_n,t_n,1,t] \
                   + (x_abc[f_n,t_n][1,0] - math.sqrt(3)*r_abc[f_n,t_n][1,0])*m.q_line[f_n,t_n,0,t] \
                   + (x_abc[f_n,t_n][1,2] + math.sqrt(3)*r_abc[f_n,t_n][1,2])*m.q_line[f_n,t_n,2,t]) == 0
        
        else:
            
            return m.voltage[t_n,p,t] - m.voltage[f_n,p,t] \
                -(-2*r_abc[f_n,t_n][2,2]*m.p_line[f_n,t_n,2,t] \
                   + (r_abc[f_n,t_n][2,0] - math.sqrt(3)*x_abc[f_n,t_n][2,0])*m.p_line[f_n,t_n,0,t] \
                   + (r_abc[f_n,t_n][2,1] + math.sqrt(3)*x_abc[f_n,t_n][2,1])*m.p_line[f_n,t_n,1,t])\
                 -(-2*x_abc[f_n,t_n][2,2]*m.q_line[f_n,t_n,2,t] \
                   + (x_abc[f_n,t_n][2,0] + math.sqrt(3)*r_abc[f_n,t_n][2,0])*m.q_line[f_n,t_n,0,t] \
                   + (x_abc[f_n,t_n][2,1] - math.sqrt(3)*r_abc[f_n,t_n][2,1])*m.q_line[f_n,t_n,1,t]) == 0
    
    def kirch_active_t(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
                    
            if  to_n == n:
                sum_1 += m.p_line[from_n,to_n,p,t]     
                
        return m.kirchoff_active_to[n,p,t] == sum_1
    
    def kirch_active_f(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
            
            if  from_n == n:
                sum_1 += m.p_line[from_n,to_n,p,t]           
                
        return m.kirchoff_active_from[n,p,t] == sum_1
    
    def kirch_active_node_eq(m, n, p, t):
        
        return m.kirchoff_active_to[n,p,t] + m.p_gen[n,p,t] + m.p_der[n,p,t] - m.kirchoff_active_from[n,p,t] - m.p_load[n,p,t] == 0
    
    def kirch_reactive_t(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
                    
            if  to_n == n:
                sum_1 += m.q_line[from_n,to_n,p,t]     
                
        return m.kirchoff_reactive_to[n,p,t] == sum_1
    
    def kirch_reactive_f(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
            
            if  from_n == n:
                sum_1 += m.q_line[from_n,to_n,p,t]           
                
        return m.kirchoff_reactive_from[n,p,t] == sum_1
    
    def kirch_reactive_node_eq(m, n, p, t):
        
        return m.kirchoff_reactive_to[n,p,t] + m.q_gen[n,p,t] + m.q_der[n,p,t] - m.kirchoff_reactive_from[n,p,t] - m.q_load[n,p,t] == 0
        
    r_abc, x_abc, g_abc, b_abc = imp_matrix.impedance_matrix(network)
    
    phases = [0,1,2]
    buses = network.bus.index
    
    from_nodes = np.concatenate((network.line.from_bus.values, network.trafo.hv_bus.values))
    to_nodes = np.concatenate((network.line.to_bus.values, network.trafo.lv_bus.values))
    
    from_nodes_l = network.line.from_bus.values
    to_nodes_l = network.line.to_bus.values
    
    from_nodes_t = network.trafo.hv_bus.values
    to_nodes_t = network.trafo.lv_bus.values
    
    times_range = list(range(0, 1))
    
    ft_list = []
    tf_list = []
    
    for i in range(0, len(from_nodes)):
        ft_pair = from_nodes[i], to_nodes[i]
        ft_list.append(ft_pair)
        
        tf_pair = [to_nodes[i], from_nodes[i]]
        tf_list.append(tf_pair)
    
    ft_list_l = []
    tf_list_l = []
    
    for i in range(0, len(from_nodes_l)):
        ft_pair = from_nodes_l[i], to_nodes_l[i]
        ft_list_l.append(ft_pair)
        
        tf_pair = [to_nodes_l[i], from_nodes_l[i]]
        tf_list_l.append(tf_pair)
    
    ft_list_t = []
    tf_list_t = []
    
    for i in range(0, len(from_nodes_t)):
        ft_pair = from_nodes_t[i], to_nodes_t[i]
        ft_list_t.append(ft_pair)
        
        tf_pair = [to_nodes_t[i], from_nodes_t[i]]
        tf_list_t.append(tf_pair)
    
    v_a = np.zeros([len(times_range), len(network.bus.index)])
    v_b = np.zeros([len(times_range), len(network.bus.index)])
    v_c = np.zeros([len(times_range), len(network.bus.index)])
    
    v_a_df = pd.DataFrame(v_a, columns = network.bus.index)
    v_b_df = pd.DataFrame(v_b, columns = network.bus.index)
    v_c_df = pd.DataFrame(v_c, columns = network.bus.index)
    
    s_a = np.zeros([len(times_range), len(network.line.index)])
    s_b = np.zeros([len(times_range), len(network.line.index)])
    s_c = np.zeros([len(times_range), len(network.line.index)])
    
    s_a_df = pd.DataFrame(s_a, columns = network.line.index)
    s_b_df = pd.DataFrame(s_b, columns = network.line.index)
    s_c_df = pd.DataFrame(s_c, columns = network.line.index)
    
    p_der_a = np.zeros([len(times_range), len(network.bus.index)])
    p_der_b = np.zeros([len(times_range), len(network.bus.index)])
    p_der_c = np.zeros([len(times_range), len(network.bus.index)])
    
    p_der_a_df = pd.DataFrame(p_der_a, columns = network.bus.index)
    p_der_b_df = pd.DataFrame(p_der_b, columns = network.bus.index)
    p_der_c_df = pd.DataFrame(p_der_c, columns = network.bus.index)
    
    objective_value = 0
    total_time = 0
    
    for t in times_range:
        print(t)
        times = [t]
        
        start = timer()
        model = pyo.ConcreteModel()    
    
        model.voltage = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 1.0)
        
        model.p_line = pyo.Var(ft_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
        model.q_line = pyo.Var(ft_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
        
        model.p_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (0,100), initialize = 0.0)
        model.q_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (0,100), initialize = 0.0)
        
        model.p_der = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (0,100), initialize = 0.0)
        model.q_der = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (0,0), initialize = 0.0)
        
        model.p_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        model.q_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        
        model.kirchoff_active_to = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        model.kirchoff_active_from = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        
        model.kirchoff_reactive_to = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        model.kirchoff_reactive_from = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        
        for t in times:
            for i in buses:
                for p in phases:
                    
                    if i == 0:
                        model.voltage[i,p,t].fix(vm_pu)
            
                if i in network.asymmetric_load.bus.values:
                    
                    model.p_load[i, 0, t].fix(load_curve_a.loc[t,i]/1e6)
                    model.p_load[i, 1, t].fix(load_curve_b.loc[t,i]/1e6)
                    model.p_load[i, 2, t].fix(load_curve_c.loc[t,i]/1e6)
        
                    model.q_load[i, 0, t].fix(load_curve_a.loc[t,i]/1e6*math.tan(math.acos(0.95)))
                    model.q_load[i, 1, t].fix(load_curve_b.loc[t,i]/1e6*math.tan(math.acos(0.95)))
                    model.q_load[i, 2, t].fix(load_curve_c.loc[t,i]/1e6*math.tan(math.acos(0.95)))
                    
                    if der_a.loc[t,i] == 1 and der_b.loc[t,i] == 0 and der_c.loc[t,i] == 0:
                        model.p_der[i, 0, t].value = 0.0
                        model.p_der[i, 1, t].fix(0.0)
                        model.p_der[i, 2, t].fix(0.0)
                    
                    elif der_a.loc[t,i] == 0 and der_b.loc[t,i] == 1 and der_c.loc[t,i] == 0:
                        model.p_der[i, 0, t].fix(0.0)
                        model.p_der[i, 1, t].value = 0.0
                        model.p_der[i, 2, t].fix(0.0)
                    
                    if der_a.loc[t,i] == 0 and der_b.loc[t,i] == 0 and der_c.loc[t,i] == 1:
                        model.p_der[i, 0, t].fix(0.0)
                        model.p_der[i, 1, t].fix(0.0)
                        model.p_der[i, 2, t].value = 0.0
                    
                    for p in phases:
                        # model.p_der[i,p,t].fix(0.0)
                        model.q_der[i,p,t].fix(0.0)
                        
                        model.p_gen[i,p,t].fix(0.0)
                        model.q_gen[i,p,t].fix(0.0)
                
                elif i not in network.asymmetric_load.bus.values:
                
                    for p in phases:
                        
                        model.p_load[i, p, t].fix(0.0)
                        model.q_load[i, p, t].fix(0.0)
                        
                        model.p_der[i, p, t].fix(0.0)
                        model.q_der[i, p, t].fix(0.0)
                                                               
                        if i!= 0:
                            
                            model.p_gen[i, p, t].fix(0.0)
                            model.q_gen[i, p, t].fix(0.0)
            
        model.const_v_magnitude_lb = pyo.Constraint(buses, phases, times, rule = voltage_magnitude_lb)
        model.const_v_magnitude_ub = pyo.Constraint(buses, phases, times, rule = voltage_magnitude_ub)
        
        model.const_v_drop_ft = pyo.Constraint(ft_list, phases, times, rule = voltage_drop_ft)
        
        model.const_kaf = pyo.Constraint(buses, phases, times, rule = kirch_active_f) #Kirchoff acitve from
        model.const_kat = pyo.Constraint(buses, phases, times, rule = kirch_active_t) #Kirchoff active to
        model.const_kirchoff_active_eq = pyo.Constraint(buses, phases, times, rule = kirch_active_node_eq)
                    
        model.const_krf = pyo.Constraint(buses, phases, times, rule = kirch_reactive_f) #Kirchoff reacitve from
        model.const_krt = pyo.Constraint(buses, phases, times, rule = kirch_reactive_t) #Kirchoff reactive to
        model.const_kirchoff_reactive_eq = pyo.Constraint(buses, phases, times, rule = kirch_reactive_node_eq)
        
        model.obj = pyo.Objective(expr = sum(model.p_der[n,p,t] for n in network.asymmetric_load.bus.values for p in phases for t in times),\
                                  sense = pyo.maximize)
        pyo.SolverFactory('knitro').solve(model, tee=True)

        
        end = timer ()
        
        objective_value += pyo.value(model.obj)
        total_time += (end-start)
        
        for i in model.voltage:
            if i[1] == 0:
                v_a_df.iloc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage[i]))
            elif i[1] == 1:
                v_b_df.iloc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage[i]))
            elif i[1] == 2:
                v_c_df.iloc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage[i]))
        
        col = 0
        col_name = []
        for i in model.p_line:
            print(i)
            if i[2] == 0:
                col_name.append((i[0], i[1]))
                s_a_df.iloc[i[3], col] = math.sqrt((pyo.value(model.p_line[i]))**2 + (pyo.value(model.q_line[i]))**2)
            elif i[2] == 1:
                s_b_df.iloc[i[3], col] = math.sqrt((pyo.value(model.p_line[i]))**2 + (pyo.value(model.q_line[i]))**2)
            elif i[2] == 2:
                s_c_df.iloc[i[3], col] = math.sqrt((pyo.value(model.p_line[i]))**2 + (pyo.value(model.q_line[i]))**2)
            
        s_a_df.columns = col_name
        s_b_df.columns = col_name
        s_c_df.columns = col_name

        for i in model.p_der:
            
            if i[1] == 0:
                p_der_a_df.iloc[i[2],i[0]] = pyo.value(model.p_der[i])
            if i[1] == 1:
                p_der_b_df.iloc[i[2],i[0]] = pyo.value(model.p_der[i])
            if i[1] == 2:
                p_der_c_df.iloc[i[2],i[0]] = pyo.value(model.p_der[i])

    print(1000*objective_value)
    print(total_time)
                       
    return objective_value, total_time, v_a_df, v_b_df, v_c_df, p_der_a_df, p_der_b_df, p_der_c_df, s_a_df, s_b_df, s_c_df