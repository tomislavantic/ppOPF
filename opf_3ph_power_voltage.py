import imp_matrix
import pyomo.environ as pyo 
import math
import numpy as np
import pandas as pd

def opf_3ph_power_voltage(network, load_curve_a, load_curve_b, load_curve_c, vm_pu):
    
    def voltage_magnitude_lb(m, n, p, t):
        return 0.81 <= m.voltage_re[n,p,t]*m.voltage_re[n,p,t] + m.voltage_im[n,p,t]*m.voltage_im[n,p,t]

    def voltage_magnitude_ub(m, n, p,t):
        return m.voltage_re[n,p,t]*m.voltage_re[n,p,t] + m.voltage_im[n,p,t]*m.voltage_im[n,p,t] <= 1.21
    
    def active_ft(m, f_n, t_n, p, t):
        
        return m.active_line_ft[f_n, t_n, p, t] == sum((m.voltage_re[f_n,p,t]*m.voltage_re[f_n,q,t] \
                                              + m.voltage_im[f_n,p,t]*m.voltage_im[f_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) +\
                                        sum((m.voltage_im[f_n,p,t]*m.voltage_re[f_n,q,t] \
                                              - m.voltage_re[f_n,p,t]*m.voltage_im[f_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases) -\
                                        sum((m.voltage_re[f_n,p,t]*m.voltage_re[t_n,q,t] \
                                              + m.voltage_im[f_n,p,t]*m.voltage_im[t_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) -\
                                        sum((m.voltage_im[f_n,p,t]*m.voltage_re[t_n,q,t] \
                                            - m.voltage_re[f_n,p,t]*m.voltage_im[t_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases)
    def active_tf(m, f_n, t_n, p, t):
        
        return m.active_line_tf[t_n, f_n, p, t] == sum((m.voltage_re[t_n,p,t]*m.voltage_re[t_n,q,t] \
                                              + m.voltage_im[t_n,p,t]*m.voltage_im[t_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) +\
                                        sum((m.voltage_im[t_n,p,t]*m.voltage_re[t_n,q,t] \
                                              - m.voltage_re[t_n,p,t]*m.voltage_im[t_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases) -\
                                        sum((m.voltage_re[t_n,p,t]*m.voltage_re[f_n,q,t] \
                                              + m.voltage_im[t_n,p,t]*m.voltage_im[f_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) -\
                                        sum((m.voltage_im[t_n,p,t]*m.voltage_re[f_n,q,t] \
                                            - m.voltage_re[t_n,p,t]*m.voltage_im[f_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases)

    def reactive_ft(m, f_n, t_n, p, t):
            
        return m.reactive_line_ft[f_n, t_n, p, t] == -sum((m.voltage_re[f_n,p,t]*m.voltage_re[f_n,q,t] \
                                              + m.voltage_im[f_n,p,t]*m.voltage_im[f_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases) +\
                                        sum((m.voltage_im[f_n,p,t]*m.voltage_re[f_n,q,t] \
                                              - m.voltage_re[f_n,p,t]*m.voltage_im[f_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) +\
                                        sum((m.voltage_re[f_n,p,t]*m.voltage_re[t_n,q,t] \
                                              + m.voltage_im[f_n,p,t]*m.voltage_im[t_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases) -\
                                        sum((m.voltage_im[f_n,p,t]*m.voltage_re[t_n,q,t] \
                                            - m.voltage_re[f_n,p,t]*m.voltage_im[t_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases)

    def reactive_tf(m, f_n, t_n, p, t):
            
        return m.reactive_line_tf[t_n, f_n, p, t] == -sum((m.voltage_re[t_n,p,t]*m.voltage_re[t_n,q,t] \
                                              + m.voltage_im[t_n,p,t]*m.voltage_im[t_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases) +\
                                        sum((m.voltage_im[t_n,p,t]*m.voltage_re[t_n,q,t] \
                                              - m.voltage_re[t_n,p,t]*m.voltage_im[t_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases) +\
                                        sum((m.voltage_re[t_n,p,t]*m.voltage_re[f_n,q,t] \
                                              + m.voltage_im[t_n,p,t]*m.voltage_im[f_n,q,t]) * b_abc[f_n,t_n][p,q] for q in phases) -\
                                        sum((m.voltage_im[t_n,p,t]*m.voltage_re[f_n,q,t] \
                                            - m.voltage_re[t_n,p,t]*m.voltage_im[f_n,q,t]) * g_abc[f_n,t_n][p,q] for q in phases)
    
    def power_flow_line_ft(m,f_n,t_n,p,t):
        #Base power = 1e6, calculating power from current
        for ft in range(0, len(network.line.index)):
            if f_n == network.line.from_bus[ft] and t_n == network.line.to_bus[ft]:
                max_current = network.line.max_i_ka[ft]
                break
        
        return m.active_line_ft[f_n,t_n,p,t]*m.active_line_ft[f_n,t_n,p,t] + \
               m.reactive_line_ft[f_n,t_n,p,t]*m.reactive_line_ft[f_n,t_n,p,t] <= (max_current*network.bus.vn_kv[f_n])**2
    
    def power_flow_line_tf(m,f_n,t_n,p,t):
        #Base power = 1e6, nominal voltage = 400/sqrt(3), Base Current = S/U
        for ft in range(0, len(network.line.index)):
            if f_n == network.line.from_bus[ft] and t_n == network.line.to_bus[ft]:
                max_current = network.line.max_i_ka[ft]
                break
        
        return m.active_line_tf[t_n,f_n,p,t]*m.active_line_tf[t_n,f_n,p,t] + \
               m.reactive_line_tf[t_n,f_n,p,t]*m.reactive_line_tf[t_n,f_n,p,t] <= (max_current*network.bus.vn_kv[t_n])**2
    
    def power_flow_trafo_ft(m,f_n,t_n,p,t):
        #Trafo is defined with maximum power, base power = 1e6
       
        for ft in range(0, len(network.trafo.index)):
            if f_n == network.trafo.hv_bus[ft] and t_n == network.trafo.lv_bus[ft]:
                max_power = network.trafo.sn_mva[ft]
                break
            
            return m.active_line_ft[f_n,t_n,p,t]*m.active_line_ft[f_n,t_n,p,t] + \
                   m.reactive_line_ft[f_n,t_n,p,t]*m.reactive_line_ft[f_n,t_n,p,t] <= max_power**2
                   
    def power_flow_trafo_tf(m,f_n,t_n,p,t):
        #Base power = 1e6, nominal voltage = 400/sqrt(3), Base Current = S/U
        for ft in range(0, len(network.trafo.index)):
            if f_n == network.trafo.hv_bus[ft] and t_n == network.trafo.lv_bus[ft]:
                max_power = network.trafo.sn_mva[ft]
                break
                    
        return m.active_line_tf[t_n,f_n,p,t]*m.active_line_tf[t_n,f_n,p,t] + \
               m.reactive_line_tf[t_n,f_n,p,t]*m.reactive_line_tf[t_n,f_n,p,t] <= max_power**2
    
    def kirch_active_t(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
                    
            if  to_n == n:
                sum_1 += m.active_line_ft[from_n,to_n,p,t] - (m.active_line_ft[from_n,to_n,p,t] + m.active_line_tf[to_n,from_n,p,t])      
                
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
        if n!=0:
            return m.kirchoff_active_to[n,p,t] + m.p_gen[n,p,t] - m.kirchoff_active_from[n,p,t] - m.p_load[n,p,t] == 0
        else:
            return m.kirchoff_active_to[n,p,t] + m.p_gen[n,p,t] - m.kirchoff_active_from[n,p,t] == 0
    
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
        if n!= 0:
            return m.kirchoff_reactive_to[n,p,t] + m.q_gen[n,p,t] - m.kirchoff_reactive_from[n,p,t] - m.q_load[n,p,t] == 0
        else:
            return m.kirchoff_reactive_to[n,p,t] + m.q_gen[n,p,t] - m.kirchoff_reactive_from[n,p,t] == 0
    
    model = pyo.ConcreteModel()
    
    r_abc, x_abc, g_abc, b_abc = imp_matrix.impedance_matrix(network)
    
    phases = [0,1,2]
    buses = network.bus.index
    
    from_nodes = network.line.from_bus
    to_nodes = network.line.to_bus
    
    ft_list = []
    tf_list = []
    
    times = list(range(0, load_curve_a.shape[0]))
    
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
    
    model.voltage_re  = pyo.Var(buses,phases, times, within = pyo.Reals)
    model.voltage_im  = pyo.Var(buses, phases, times, within = pyo.Reals)
    
    model.active_line_ft = pyo.Var(ft_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
    model.reactive_line_ft = pyo.Var(ft_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
    
    model.active_line_tf = pyo.Var(tf_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
    model.reactive_line_tf = pyo.Var(tf_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
    
    model.p_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
    model.q_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
    
    model.p_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
    model.q_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
    
    model.kirchoff_active_to = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
    model.kirchoff_active_from = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
    
    model.kirchoff_reactive_to = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
    model.kirchoff_reactive_from = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
    
    model.x_single_phase_pv = pyo.Var(network.asymmetric_load.bus.values, phases, times, domain = pyo.Binary, initialize = 0.0)
    model.x_three_phase_pv = pyo.Var(network.asymmetric_load.bus.values, times, domain = pyo.Binary, initialize = 0.0)
    
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
        
        for i in range(0, len(network.asymmetric_load.bus)):
            model.p_load[network.asymmetric_load.bus[i], 0, t].fix(load_curve_a.iloc[t,i]/1e6)
            model.p_load[network.asymmetric_load.bus[i], 1, t].fix(load_curve_b.iloc[t,i]/1e6)
            model.p_load[network.asymmetric_load.bus[i], 2, t].fix(load_curve_c.iloc[t,i]/1e6)
            
            model.q_load[network.asymmetric_load.bus[i], 0, t].fix(load_curve_a.iloc[t,i]*math.tan(math.acos(0.95))/1e6)
            model.q_load[network.asymmetric_load.bus[i], 1, t].fix(load_curve_b.iloc[t,i]*math.tan(math.acos(0.95))/1e6)
            model.q_load[network.asymmetric_load.bus[i], 2, t].fix(load_curve_c.iloc[t,i]*math.tan(math.acos(0.95))/1e6)
            
            model.p_gen[network.asymmetric_load.bus[i], 0, t].fix(0.0)
            model.p_gen[network.asymmetric_load.bus[i], 1, t].fix(0.0)
            model.p_gen[network.asymmetric_load.bus[i], 2, t].fix(0.0)
            
            model.q_gen[network.asymmetric_load.bus[i], 0, t].fix(0.0)
            model.q_gen[network.asymmetric_load.bus[i], 1, t].fix(0.0)
            model.q_gen[network.asymmetric_load.bus[i], 2, t].fix(0.0)
    
        for i in buses:
            
            if i not in network.asymmetric_load.bus.values and i!=0:
                for p in phases:
                    model.p_load[i, p, t].fix(0.0)
                    model.q_load[i, p, t].fix(0.0)
                                   
                    model.p_gen[i, p, t].fix(0.0)
                    model.q_gen[i, p, t].fix(0.0)
    
    model.const_v_magnitude_lb = pyo.Constraint(buses, phases, times, rule = voltage_magnitude_lb)
    model.const_v_magnitude_ub = pyo.Constraint(buses, phases, times, rule = voltage_magnitude_ub)
    
    model.const_active_ft = pyo.Constraint(ft_list, phases, times, rule = active_ft)
    model.const_active_tf = pyo.Constraint(ft_list, phases, times, rule = active_tf)
    model.const_reactive_ft = pyo.Constraint(ft_list, phases, times, rule = reactive_ft)
    model.const_reactive_tf = pyo.Constraint(ft_list, phases, times, rule = reactive_tf)
    
    model.const_power_flow_line_ft = pyo.Constraint(ft_list_l, phases, times, rule = power_flow_line_ft)
    model.const_power_flow_line_tf = pyo.Constraint(ft_list_l, phases, times, rule = power_flow_line_tf)
    
    model.const_power_flow_trafo_ft = pyo.Constraint(ft_list_t, phases, times, rule = power_flow_trafo_ft)
    model.const_power_flow_trafo_tf = pyo.Constraint(ft_list_t, phases, times, rule = power_flow_trafo_tf)
    
    model.const_kaf = pyo.Constraint(buses, phases, times, rule = kirch_active_f) #Kirchoff acitve from
    model.const_kat = pyo.Constraint(buses, phases, times, rule = kirch_active_t) #Kirchoff active to
    model.const_kirchoff_active_eq = pyo.Constraint(buses, phases, times, rule = kirch_active_node_eq)
    
    model.const_krf = pyo.Constraint(buses, phases, times, rule = kirch_reactive_f) #Kirchoff reacitve from
    model.const_krt = pyo.Constraint(buses, phases, times, rule = kirch_reactive_t) #Kirchoff reactive to
    model.const_kirchoff_reactive_eq = pyo.Constraint(buses, phases, times, rule = kirch_reactive_node_eq)
    
    model.obj = pyo.Objective(expr = sum(model.p_gen[0,0,t] + model.p_gen[0,1,t] + model.p_gen[0,2,t] for t in times), sense = pyo.minimize)
        
    pyo.SolverFactory('ipopt').solve(model, tee = True) 
    
    print(pyo.value(model.obj))
    
    v_a = np.zeros([len(times), len(network.bus.index)])
    v_b = np.zeros([len(times), len(network.bus.index)])
    v_c = np.zeros([len(times), len(network.bus.index)])
    
    v_a_df = pd.DataFrame(v_a, columns = network.bus.index)
    v_b_df = pd.DataFrame(v_b, columns = network.bus.index)
    v_c_df = pd.DataFrame(v_c, columns = network.bus.index)
        
    for i in model.voltage_re:
        if i[1] == 0:
            v_a_df.loc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
        elif i[1] == 1:
            v_b_df.loc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
        else:
            v_c_df.loc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
    
    
    return v_a_df, v_b_df, v_c_df
