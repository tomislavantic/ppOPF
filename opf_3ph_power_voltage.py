import imp_matrix
import pyomo.environ as pyo 
import math
import numpy as np
import pandas as pd
from timeit import default_timer as timer

def opf_3ph_power_voltage(network, load_curve_a, load_curve_b, load_curve_c, der_a, der_b, der_c, vm_pu):
    
    def voltage_magnitude_lb(m, n, p, t):
        return 0.81 <= m.voltage_re[n,p,t]*m.voltage_re[n,p,t] + m.voltage_im[n,p,t]*m.voltage_im[n,p,t]

    def voltage_magnitude_ub(m, n, p,t):
        return m.voltage_re[n,p,t]*m.voltage_re[n,p,t] + m.voltage_im[n,p,t]*m.voltage_im[n,p,t] <= 1.21
    
    def vuf_limit(m,n,t):
        
        return  4e-4*((m.voltage_re[n,0,t] - 0.5*(m.voltage_re[n,1,t] + m.voltage_re[n,2,t]) - \
                          math.sqrt(3)/2*(m.voltage_im[n,1,t] - m.voltage_im[n,2,t]))*\
                         (m.voltage_re[n,0,t] - 0.5*(m.voltage_re[n,1,t] + m.voltage_re[n,2,t]) - \
                          math.sqrt(3)/2*(m.voltage_im[n,1,t] - m.voltage_im[n,2,t])) + \
                         (m.voltage_im[n,0,t] + math.sqrt(3)/2*(m.voltage_re[n,1,t] - m.voltage_re[n,2,t]) - \
                          0.5*(m.voltage_im[n,1,t] + m.voltage_im[n,2,t]))*\
                         (m.voltage_im[n,0,t] + math.sqrt(3)/2*(m.voltage_re[n,1,t] - m.voltage_re[n,2,t]) - \
                          0.5*(m.voltage_im[n,1,t] + m.voltage_im[n,2,t]))) >= \
                                                                                \
                        ((m.voltage_re[n,0,t] - 0.5*(m.voltage_re[n,1,t] + m.voltage_re[n,2,t]) + \
                         math.sqrt(3)/2*(m.voltage_im[n,1,t] - m.voltage_im[n,2,t]))*\
                        (m.voltage_re[n,0,t] - 0.5*(m.voltage_re[n,1,t] + m.voltage_re[n,2,t]) + \
                         math.sqrt(3)/2*(m.voltage_im[n,1,t] - m.voltage_im[n,2,t])) + \
                        (m.voltage_im[n,0,t] - math.sqrt(3)/2*(m.voltage_re[n,1,t] - m.voltage_re[n,2,t]) - \
                         0.5*(m.voltage_im[n,1,t] + m.voltage_im[n,2,t]))*\
                        (m.voltage_im[n,0,t] - math.sqrt(3)/2*(m.voltage_re[n,1,t] - m.voltage_re[n,2,t]) - \
                         0.5*(m.voltage_im[n,1,t] + m.voltage_im[n,2,t])))
    
    def active_ft(m, f_n, t_n, p, t):
        
        return m.active_line_ft[f_n,t_n,p,t] == \
            sum((m.voltage_re[f_n,p,t]*m.voltage_re[f_n,q,t] + m.voltage_im[f_n,p,t]*m.voltage_im[f_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) +\
            sum((m.voltage_im[f_n,p,t]*m.voltage_re[f_n,q,t] - m.voltage_re[f_n,p,t]*m.voltage_im[f_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases) -\
            sum((m.voltage_re[f_n,p,t]*m.voltage_re[t_n,q,t] + m.voltage_im[f_n,p,t]*m.voltage_im[t_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) -\
            sum((m.voltage_im[f_n,p,t]*m.voltage_re[t_n,q,t] - m.voltage_re[f_n,p,t]*m.voltage_im[t_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases)
        
    def reactive_ft(m, f_n, t_n, p, t):
         
        return m.reactive_line_ft[f_n,t_n,p,t] ==\
            -sum(((m.voltage_re[f_n,p,t]*m.voltage_re[f_n,q,t] + m.voltage_im[f_n,p,t]*m.voltage_im[f_n,q,t]) * b_abc[f_n,t_n][p,q]) for q in phases) +\
            sum(((m.voltage_im[f_n,p,t]*m.voltage_re[f_n,q,t] - m.voltage_re[f_n,p,t]*m.voltage_im[f_n,q,t]) * g_abc[f_n,t_n][p,q]) for q in phases) +\
            sum(((m.voltage_re[f_n,p,t]*m.voltage_re[t_n,q,t] + m.voltage_im[f_n,p,t]*m.voltage_im[t_n,q,t]) * b_abc[f_n,t_n][p,q]) for q in phases) -\
            sum(((m.voltage_im[f_n,p,t]*m.voltage_re[t_n,q,t] - m.voltage_re[f_n,p,t]*m.voltage_im[t_n,q,t]) * g_abc[f_n,t_n][p,q]) for q in phases)
    
    def active_tf(m, f_n, t_n, p, t):
        
        return m.active_line_tf[t_n,f_n,p,t] == \
            sum((m.voltage_re[t_n,p,t]*m.voltage_re[t_n,q,t] + m.voltage_im[t_n,p,t]*m.voltage_im[t_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) +\
            sum((m.voltage_im[t_n,p,t]*m.voltage_re[t_n,q,t] - m.voltage_re[t_n,p,t]*m.voltage_im[t_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases) -\
            sum((m.voltage_re[t_n,p,t]*m.voltage_re[f_n,q,t] + m.voltage_im[t_n,p,t]*m.voltage_im[f_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) -\
            sum((m.voltage_im[t_n,p,t]*m.voltage_re[f_n,q,t] - m.voltage_re[t_n,p,t]*m.voltage_im[f_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases)
        
    def reactive_tf(m, f_n, t_n, p, t):
          
        return m.reactive_line_tf[t_n,f_n,p,t] ==\
           -sum(((m.voltage_re[t_n,p,t]*m.voltage_re[t_n,q,t] + m.voltage_im[t_n,p,t]*m.voltage_im[t_n,q,t]) * b_abc[f_n,t_n][p,q]) for q in phases) +\
            sum(((m.voltage_im[t_n,p,t]*m.voltage_re[t_n,q,t] - m.voltage_re[t_n,p,t]*m.voltage_im[t_n,q,t]) * g_abc[f_n,t_n][p,q]) for q in phases) +\
            sum(((m.voltage_re[t_n,p,t]*m.voltage_re[f_n,q,t] + m.voltage_im[t_n,p,t]*m.voltage_im[f_n,q,t]) * b_abc[f_n,t_n][p,q]) for q in phases) -\
            sum(((m.voltage_im[t_n,p,t]*m.voltage_re[f_n,q,t] - m.voltage_re[t_n,p,t]*m.voltage_im[f_n,q,t]) * g_abc[f_n,t_n][p,q]) for q in phases)

   
    def power_flow_line_ft(m,f_n,t_n,p,t):
        #Base power = 1e6, calculating power from current
        for ft in range(0, len(network.line.index)):
            if f_n == network.line.from_bus[ft] and t_n == network.line.to_bus[ft]:
                max_current = network.line.max_i_ka[ft]
                break
        
        return m.active_line_ft[f_n,t_n,p,t]*m.active_line_ft[f_n,t_n,p,t] + \
               m.reactive_line_ft[f_n,t_n,p,t]*m.reactive_line_ft[f_n,t_n,p,t] <= (max_current*network.bus.vn_kv[f_n])**2
    
    
    def power_flow_trafo_ft(m,f_n,t_n,p,t):
        #Trafo is defined with maximum power, base power = 1e6
       
        for ft in range(0, len(network.trafo.index)):
            if f_n == network.trafo.hv_bus[ft] and t_n == network.trafo.lv_bus[ft]:
                max_power = network.trafo.sn_mva[ft]
                break
            
        return m.active_line_ft[f_n,t_n,p,t]*m.active_line_ft[f_n,t_n,p,t] + \
               m.reactive_line_ft[f_n,t_n,p,t]*m.reactive_line_ft[f_n,t_n,p,t] <= max_power**2
   
    def kirch_active_t(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
                    
            if  to_n == n:
                sum_1 += m.active_line_ft[from_n,to_n,p,t]
                
        return m.kirchoff_active_to[n,p,t] == sum_1
    
    def kirch_active_f(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
            
            if  from_n == n:
                sum_1 += m.active_line_ft[from_n,to_n,p,t]           
                
        return m.kirchoff_active_from[n,p,t] == sum_1
    
    def kirch_active_node_eq(m, n, p, t):
        
        return m.kirchoff_active_to[n,p,t] + m.p_gen[n,p,t] + m.p_der[n,p,t] - m.kirchoff_active_from[n,p,t] - m.p_load[n,p,t] == 0
    
    def kirch_reactive_t(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
                    
            if  to_n == n:
                sum_1 += m.reactive_line_ft[from_n,to_n,p,t]
                
        return m.kirchoff_reactive_to[n,p,t] == sum_1
    
    def kirch_reactive_f(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
            
            if  from_n == n:
                sum_1 += m.reactive_line_ft[from_n,to_n,p,t]           
                
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
    
        model.voltage_re  = pyo.Var(buses,phases, times, within = pyo.Reals, bounds = (-10,10))
        model.voltage_im  = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        
        model.active_line_ft = pyo.Var(ft_list, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        model.reactive_line_ft = pyo.Var(ft_list, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        
        model.active_line_tf = pyo.Var(tf_list, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        model.reactive_line_tf = pyo.Var(tf_list, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
     
        model.p_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        model.q_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        
        model.p_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        model.q_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        
        model.p_der = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (0,100), initialize = 0.0)
        model.q_der = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (0,100), initialize = 0.0)
        
        model.kirchoff_active_to = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100))
        model.kirchoff_active_from = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100))
        
        model.kirchoff_reactive_to = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100))
        model.kirchoff_reactive_from = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100))
        
        for t in times:
            for i in buses:
                if i == 0:
                    model.voltage_re[i,0,t].fix(vm_pu*math.cos(0))
                    model.voltage_re[i,1,t].fix(vm_pu*math.cos(-2/3*math.pi))
                    model.voltage_re[i,2,t].fix(vm_pu*math.cos(2/3*math.pi))
            
                    model.voltage_im[i,0,t].fix(vm_pu*math.sin(0))
                    model.voltage_im[i,1,t].fix(vm_pu*math.sin(-2/3*math.pi))
                    model.voltage_im[i,2,t].fix(vm_pu*math.sin(2/3*math.pi))
                            
                else:
                    model.voltage_re[i,0,t].value = vm_pu*math.cos(0)
                    model.voltage_re[i,1,t].value = vm_pu*math.cos(-2/3*math.pi)
                    model.voltage_re[i,2,t].value = vm_pu*math.cos(2/3*math.pi)
                    
                    model.voltage_im[i,0,t].value = vm_pu*math.sin(0)
                    model.voltage_im[i,1,t].value = vm_pu*math.sin(-2/3*math.pi)
                    model.voltage_im[i,2,t].value = vm_pu*math.sin(2/3*math.pi)
            
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
        
        model.constr_vuf = pyo.Constraint(buses, times, rule = vuf_limit)
        
        model.const_active_ft = pyo.Constraint(ft_list, phases, times, rule = active_ft)
        model.const_reactive_ft = pyo.Constraint(ft_list, phases, times, rule = reactive_ft)
        
        model.const_active_tf = pyo.Constraint(ft_list, phases, times, rule = active_tf)
        model.const_reactive_tf = pyo.Constraint(ft_list, phases, times, rule = reactive_tf)
        
        model.const_power_flow_line_ft = pyo.Constraint(ft_list_l, phases, times, rule = power_flow_line_ft)
        model.const_power_flow_trafo_ft = pyo.Constraint(ft_list_t, phases, times, rule = power_flow_trafo_ft)
        
        model.const_kaf = pyo.Constraint(buses, phases, times, rule = kirch_active_f) #Kirchoff acitve from
        model.const_kat = pyo.Constraint(buses, phases, times, rule = kirch_active_t) #Kirchoff active to
        model.const_kirchoff_active_eq = pyo.Constraint(buses, phases, times, rule = kirch_active_node_eq)
        
        model.const_krf = pyo.Constraint(buses, phases, times, rule = kirch_reactive_f) #Kirchoff reacitve from
        model.const_krt = pyo.Constraint(buses, phases, times, rule = kirch_reactive_t) #Kirchoff reactive to
        model.const_kirchoff_reactive_eq = pyo.Constraint(buses, phases, times, rule = kirch_reactive_node_eq)
        
        model.obj = pyo.Objective(expr = sum(model.p_der[n,p,t] for n in network.asymmetric_load.bus.values for p in phases for t in times),\
                                  sense = pyo.maximize)
        pyo.SolverFactory('ipopt').solve(model, tee=True) 
        
        end = timer ()
        
        objective_value += pyo.value(model.obj)
        total_time += (end-start)
        
        for i in model.voltage_re:
                if i[1] == 0:
                    v_a_df.iloc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
                elif i[1] == 1:
                    v_b_df.iloc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
                elif i[1] == 2:
                    v_c_df.iloc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
        
        col = -1
        col_name = []
        for i,j in enumerate(model.p_line):
            if i % 3 == 0:
                col_name.append((j[0], j[1]))
                col += 1
                
            if j[2] == 0:
                s_a_df.iloc[j[3], col] = math.sqrt((pyo.value(model.p_line[j]))**2 + (pyo.value(model.q_line[j]))**2)
            elif j[2] == 1:
                s_b_df.iloc[j[3], col] = math.sqrt((pyo.value(model.p_line[j]))**2 + (pyo.value(model.q_line[j]))**2)
            elif j[2] == 2:
                s_c_df.iloc[j[3], col] = math.sqrt((pyo.value(model.p_line[j]))**2 + (pyo.value(model.q_line[j]))**2)
            
        s_a_df.columns = col_name
        s_b_df.columns = col_name
        s_c_df.columns = col_name

        for i in model.p_der:
            print(i, 1000*pyo.value(model.p_load[i]))
            if i[1] == 0:
                p_der_a_df.iloc[i[2],i[0]] = pyo.value(model.p_der[i])
            if i[1] == 1:
                p_der_b_df.iloc[i[2],i[0]] = pyo.value(model.p_der[i])
            if i[1] == 2:
                p_der_c_df.iloc[i[2],i[0]] = pyo.value(model.p_der[i])

    print(1000*objective_value)
    print(total_time)
                       
    return objective_value, total_time, v_a_df, v_b_df, v_c_df, p_der_a_df, p_der_b_df, p_der_c_df, s_a_df, s_b_df, s_c_df
