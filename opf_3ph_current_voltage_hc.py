import imp_matrix
import pyomo.environ as pyo 
import math
import numpy as np
import pandas as pd
import random as rd

def opf_3ph_current_voltage(network, load_curve_a, load_curve_b, load_curve_c, pv_curve, vm_pu):
    
    def voltage_magnitude_lb(m, n, p, t):
        return 0.81 <= m.voltage_re[n,p,t]*m.voltage_re[n,p,t] + m.voltage_im[n,p,t]*m.voltage_im[n,p,t]

    def voltage_magnitude_ub(m, n, p, t):
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
    
    def voltage_drop_real_ft(m, f_n, t_n, p, t):
        
        return m.voltage_re[f_n,p,t] - m.voltage_re[t_n,p,t] - sum(r_abc[f_n,t_n][p,q]*m.current_re_line_ft[f_n,t_n,q,t] for q in phases) +\
            sum(x_abc[f_n,t_n][p,q]*m.current_im_line_ft[f_n,t_n,q,t] for q in phases) == 0
            
    def voltage_drop_real_tf(m, f_n, t_n, p, t):
        
        return m.voltage_re[t_n,p,t] - m.voltage_re[f_n,p,t] - sum(r_abc[f_n,t_n][p,q]*m.current_re_line_tf[t_n,f_n,q,t] for q in phases) +\
            sum(x_abc[f_n,t_n][p,q]*m.current_im_line_tf[t_n,f_n,q,t] for q in phases) == 0
    
    def voltage_drop_imag_ft(m, f_n, t_n, p, t):
       
        return m.voltage_im[f_n,p,t] - m.voltage_im[t_n,p,t] - sum(r_abc[f_n,t_n][p,q]*m.current_im_line_ft[f_n,t_n,q,t] for q in phases) -\
            sum(x_abc[f_n,t_n][p,q]*m.current_re_line_ft[f_n,t_n,q,t] for q in phases) == 0
    
    def voltage_drop_imag_tf(m, f_n, t_n, p, t):
        
        return m.voltage_im[t_n,p,t] - m.voltage_im[f_n,p,t] - sum(r_abc[f_n,t_n][p,q]*m.current_im_line_tf[t_n,f_n,q,t] for q in phases) -\
            sum(x_abc[f_n,t_n][p,q]*m.current_re_line_tf[t_n,f_n,q,t] for q in phases) == 0
    
    def current_flow_line_ft(m,f_n,t_n,p,t):
        #Base power = 1e6, nominal voltage = 400/sqrt(3), Base Current = S/U
        for ft in range(0, len(network.line.index)):
            if f_n == network.line.from_bus[ft] and t_n == network.line.to_bus[ft]:
                max_current = network.line.max_i_ka[ft]*1000/(network.sn_mva*1e6/(400/math.sqrt(3)))
                break
       
        return m.current_re_line_ft[f_n,t_n,p,t]*m.current_re_line_ft[f_n,t_n,p,t] + \
               m.current_im_line_ft[f_n,t_n,p,t]*m.current_im_line_ft[f_n,t_n,p,t] <= max_current**2
    
    def current_flow_line_tf(m,f_n,t_n,p,t):
        #Base power = 1e6, nominal voltage = 400/sqrt(3), Base Current = S/U
        for ft in range(0, len(network.line.index)):
            if f_n == network.line.from_bus[ft] and t_n == network.line.to_bus[ft]:
                max_current = network.line.max_i_ka[ft]*1000/(network.sn_mva*1e6/(400/math.sqrt(3)))
                break
           
        return m.current_re_line_tf[t_n,f_n,p,t]*m.current_re_line_tf[t_n,f_n,p,t] + \
               m.current_im_line_tf[t_n,f_n,p,t]*m.current_im_line_tf[t_n,f_n,p,t] <= max_current**2
    
    def current_flow_trafo_ft(m,f_n,t_n,p,t):
        #Trafo is defined with maximum power, base power = 1e6
        for ft in range(0, len(network.trafo.index)):
            if f_n == network.trafo.hv_bus[ft] and t_n == network.trafo.lv_bus[ft]:
                max_current = (network.trafo.sn_mva[ft]*1e6)/(network.trafo.vn_lv_kv[ft]*1e3/math.sqrt(3))/(network.sn_mva*1e6/(400/math.sqrt(3)))
                break   

        return m.current_re_line_ft[f_n,t_n,p,t]*m.current_re_line_ft[f_n,t_n,p,t] + \
                   m.current_im_line_ft[f_n,t_n,p,t]*m.current_im_line_ft[f_n,t_n,p,t] <= max_current**2
                   
    def current_flow_trafo_tf(m,f_n,t_n,p,t):
        #Base power = 1e6, nominal voltage = 400/sqrt(3), Base Current = S/U
        for ft in range(0, len(network.trafo.index)):
            if f_n == network.trafo.hv_bus[ft] and t_n == network.trafo.lv_bus[ft]:
                max_current = (network.trafo.sn_mva[ft]*1e6)/(network.trafo.vn_lv_kv[ft]*1e3/math.sqrt(3))/(network.sn_mva*1e6/(400/math.sqrt(3)))
                break 
            
        return m.current_re_line_tf[t_n,f_n,p,t]*m.current_re_line_tf[t_n,f_n,p,t] + \
               m.current_im_line_tf[t_n,f_n,p,t]*m.current_im_line_tf[t_n,f_n,p,t] <= max_current**2
    
    def current_gen_re(m, n, p, t):
        return m.p_gen[n, p, t] == m.voltage_re[n, p, t] * m.i_re_gen[n,p,t] + m.voltage_im[n, p, t] * m.i_im_gen[n,p,t]
    
    def current_gen_im(m, n, p, t):
        return m.q_gen[n, p, t] == m.voltage_im[n, p, t] * m.i_re_gen[n,p,t] - m.voltage_re[n, p, t] * m.i_im_gen[n,p,t]
    
    def current_load_re(m, n, p, t):
        return model.p_load[n, p, t] == m.voltage_re[n, p, t] * m.i_re_load[n,p,t] + m.voltage_im[n, p, t] * m.i_im_load[n,p,t]
    
    def current_load_im(m, n, p, t):
        return model.q_load[n, p, t] == m.voltage_im[n, p, t] * m.i_re_load[n,p,t] - m.voltage_re[n, p, t] * m.i_im_load[n,p,t]
                                              
    def kirch_current_re_t(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
                    
            if  to_n == n:
                sum_1 += m.current_re_line_ft[from_n,to_n,p,t]      
                
        return m.kirchoff_current_re_to[n,p,t] == sum_1
    
    def kirch_current_re_f(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
            
            if  from_n == n:
                sum_1 += m.current_re_line_ft[from_n,to_n,p,t]           
                
        return m.kirchoff_current_re_from[n,p,t] == sum_1
    
    def kirch_current_re_node_eq(m, n, p, t):
        if n!=0:
            return m.kirchoff_current_re_to[n,p,t] + m.i_re_gen[n,p,t] - m.kirchoff_current_re_from[n,p,t] - m.i_re_load[n,p,t] == 0
        else:
            return m.kirchoff_current_re_to[n,p,t] + m.i_re_gen[n,p,t] - m.kirchoff_current_re_from[n,p,t] == 0
    
    def kirch_current_im_t(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
                    
            if  to_n == n:
                sum_1 += m.current_im_line_ft[from_n,to_n,p,t]      
                
        return m.kirchoff_current_im_to[n,p,t] == sum_1
    
    def kirch_current_im_f(m, n, p, t):
        sum_1 = 0
        
        for x in range(0, len(from_nodes)):
            from_n = from_nodes[x]
            to_n = to_nodes[x]
            
            if  from_n == n:
                sum_1 += m.current_im_line_ft[from_n,to_n,p,t]           
                
        return m.kirchoff_current_im_from[n,p,t] == sum_1
    
    def kirch_current_im_node_eq(m, n, p, t):
        if n!= 0:
            return m.kirchoff_current_im_to[n,p,t] + m.i_im_gen[n,p,t] - m.kirchoff_current_im_from[n,p,t] - m.i_im_load[n,p,t] == 0
        else:
            return m.kirchoff_current_im_to[n,p,t] + m.i_im_gen[n,p,t] - m.kirchoff_current_im_from[n,p,t] == 0
              
    def pv_power_single_phase_ub(m,n,p,t):
        return m.p_gen[n,p,t] <= (3.68/1000) * pv_curve.iloc[t,0]
        
    r_abc, x_abc, g_abc, b_abc = imp_matrix.impedance_matrix(network)
    
    phases = [0,1,2]
    buses = network.bus.index
    
    from_nodes = np.concatenate((network.line.from_bus.values, network.trafo.hv_bus.values))
    to_nodes = np.concatenate((network.line.to_bus.values, network.trafo.lv_bus.values))
    
    from_nodes_l = network.line.from_bus.values
    to_nodes_l = network.line.to_bus.values
    
    from_nodes_t = network.trafo.hv_bus.values
    to_nodes_t = network.trafo.lv_bus.values
    
    times_range = list(range(0, load_curve_a.shape[0]))
    # times_range = list(range(0, 96))
    
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
        
    #Needed for single-phase connection of PVs or other DERs
    pv_phase = {}
    
    for i in network.asymmetric_load.bus.values:
        pv_phase[i] = rd.choice([0,1,2])
    
    v_a = np.zeros([len(times_range), len(network.bus.index)])
    v_b = np.zeros([len(times_range), len(network.bus.index)])
    v_c = np.zeros([len(times_range), len(network.bus.index)])
    
    v_a_df = pd.DataFrame(v_a, columns = network.bus.index)
    v_b_df = pd.DataFrame(v_b, columns = network.bus.index)
    v_c_df = pd.DataFrame(v_c, columns = network.bus.index)
    
    vuf = np.zeros([len(times_range), len(network.bus.index)])
    vuf_df = pd.DataFrame(vuf, columns = network.bus.index)
    
    p_gen_a = np.zeros([len(times_range), len(network.asymmetric_load.index)])
    p_gen_b = np.zeros([len(times_range), len(network.asymmetric_load.index)])
    p_gen_c = np.zeros([len(times_range), len(network.asymmetric_load.index)])
    
    p_gen_a_df = pd.DataFrame(p_gen_a, columns = network.asymmetric_load.bus)
    p_gen_b_df = pd.DataFrame(p_gen_b, columns = network.asymmetric_load.bus)
    p_gen_c_df = pd.DataFrame(p_gen_c, columns = network.asymmetric_load.bus)
                    
    for t in times_range:
        print(t)
        times = [t]
    
        model = pyo.ConcreteModel()    
        
        model.voltage_re  = pyo.Var(buses,phases, times, within = pyo.Reals, bounds = (-10,10))
        model.voltage_im  = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        
        model.current_re_line_ft = pyo.Var(ft_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
        model.current_im_line_ft = pyo.Var(ft_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
        
        model.current_re_line_tf = pyo.Var(tf_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
        model.current_im_line_tf = pyo.Var(tf_list, phases, times, within = pyo.Reals, bounds = (-10,10), initialize = 0.0)
            
        model.p_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        model.q_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        
        model.p_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (0,100), initialize = 0.0)
        model.q_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (0,100), initialize = 0.0)
        
        model.i_re_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        model.i_im_gen = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        
        model.i_re_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        model.i_im_load = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-100,100), initialize = 0.0)
        
        model.kirchoff_current_re_to = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        model.kirchoff_current_re_from = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        
        model.kirchoff_current_im_to = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
        model.kirchoff_current_im_from = pyo.Var(buses, phases, times, within = pyo.Reals, bounds = (-10,10))
            
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

                model.p_load[i, 0, t].fix(2*load_curve_a.loc[t,i]/1e6)
                model.p_load[i, 1, t].fix(2*load_curve_b.loc[t,i]/1e6)
                model.p_load[i, 2, t].fix(2*load_curve_c.loc[t,i]/1e6)
    
                model.q_load[i, 0, t].fix(2*load_curve_a.loc[t,i]/1e6*math.tan(math.acos(0.95)))
                model.q_load[i, 1, t].fix(2*load_curve_b.loc[t,i]/1e6*math.tan(math.acos(0.95)))
                model.q_load[i, 2, t].fix(2*load_curve_c.loc[t,i]/1e6*math.tan(math.acos(0.95)))
            
                #For single-phase connection with randomly defined connection phase
                if pv_phase[i] == 0:
                
                    model.p_gen[i, 0, t].value = 0.0
                    model.p_gen[i, 1, t].fix(0.0)
                    model.p_gen[i, 2, t].fix(0.0)
                    
                    model.q_gen[i, 0, t].fix(0.0)
                    model.q_gen[i, 1, t].fix(0.0)
                    model.q_gen[i, 2, t].fix(0.0)
                
                elif pv_phase[i] == 1:
                
                    model.p_gen[i, 0, t].fix(0.0)
                    model.p_gen[i, 1, t].value = 0.0
                    model.p_gen[i, 2, t].fix(0.0)
                    
                    model.q_gen[i, 0, t].fix(0.0)
                    model.q_gen[i, 1, t].fix(0.0)
                    model.q_gen[i, 2, t].fix(0.0)
                
                else:
                    model.p_gen[i, 0, t].fix(0.0)
                    model.p_gen[i, 1, t].fix(0.0)
                    model.p_gen[i, 2, t].value = 0.0
                         
                    model.q_gen[i, 0, t].fix(0.0)
                    model.q_gen[i, 1, t].fix(0.0)
                    model.q_gen[i, 2, t].fix(0.0)
        
            elif i not in network.asymmetric_load.bus.values and i!=0:
            
                for p in phases:
                    model.p_load[i, p, t].fix(0.0)
                    model.q_load[i, p, t].fix(0.0)
                                   
                    model.p_gen[i, p, t].fix(0.0)
                    model.q_gen[i, p, t].fix(0.0)
    
        model.const_v_magnitude_lb = pyo.Constraint(buses, phases, times, rule = voltage_magnitude_lb)
        model.const_v_magnitude_ub = pyo.Constraint(buses, phases, times, rule = voltage_magnitude_ub)
        
        model.const_v_drop_re_ft = pyo.Constraint(ft_list, phases, times, rule = voltage_drop_real_ft)
        model.const_v_drop_re_tf = pyo.Constraint(ft_list, phases, times, rule = voltage_drop_real_tf)
        model.const_v_drop_im_ft = pyo.Constraint(ft_list, phases, times, rule = voltage_drop_imag_ft)
        model.const_v_drop_im_tf = pyo.Constraint(ft_list, phases, times, rule = voltage_drop_imag_tf)
        
        model.const_current_flow_line_ft = pyo.Constraint(ft_list_l, phases, times, rule = current_flow_line_ft)
        model.const_current_flow_line_tf = pyo.Constraint(ft_list_l, phases, times, rule = current_flow_line_tf)
        
        model.const_current_flow_trafo_ft = pyo.Constraint(ft_list_t, phases, times, rule = current_flow_trafo_ft)
        model.const_current_flow_trafo_tf = pyo.Constraint(ft_list_t, phases, times, rule = current_flow_trafo_tf)
        
        model.const_i_load_re = pyo.Constraint(buses, phases, times, rule = current_load_re)
        model.const_i_load_im = pyo.Constraint(buses, phases, times, rule = current_load_im)
        
        model.const_i_gen_re = pyo.Constraint(buses, phases, times, rule = current_gen_re)
        model.const_i_gen_im = pyo.Constraint(buses, phases, times, rule = current_gen_im)
        
        model.const_k_i_re_f = pyo.Constraint(buses, phases, times, rule = kirch_current_re_f) #Kirchoff acitve from
        model.const_k_i_re_t = pyo.Constraint(buses, phases, times, rule = kirch_current_re_t) #Kirchoff active to
        model.const_kirchoff_i_re_eq = pyo.Constraint(buses, phases, times, rule = kirch_current_re_node_eq)
        
        model.const_k_i_im_f = pyo.Constraint(buses, phases, times, rule = kirch_current_im_f) #Kirchoff acitve from
        model.const_k_i_im_t = pyo.Constraint(buses, phases, times, rule = kirch_current_im_t) #Kirchoff active to
        model.const_kirchoff_i_im_eq = pyo.Constraint(buses, phases, times, rule = kirch_current_im_node_eq)
        
        model.constr_vuf = pyo.Constraint(buses, times, rule = vuf_limit)
               
        model.constr_pv_single_phase_pv_ub = pyo.Constraint(network.asymmetric_load.bus.values, phases, times, rule = pv_power_single_phase_ub)
        
        model.obj = pyo.Objective(expr = sum(model.p_gen[n,p,t] for n in network.asymmetric_load.bus.values for p in phases for t in times), sense = pyo.maximize)
                
        pyo.SolverFactory('ipopt').solve(model, tee=True) 
        
        for i in model.voltage_re:
            if i[1] == 0:
                v_a_df.loc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
            elif i[1] == 1:
                v_b_df.loc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
            else:
                v_c_df.loc[i[2], i[0]] = math.sqrt(pyo.value(model.voltage_re[i])**2 + pyo.value(model.voltage_im[i])**2)
            
        for n in network.bus.index:                             
            
            vuf_df.iloc[t,n] = math.sqrt(((pyo.value(model.voltage_re[n,0,t]) - 0.5*(pyo.value(model.voltage_re[n,1,t]) + pyo.value(model.voltage_re[n,2,t])) + \
              math.sqrt(3)/2*(pyo.value(model.voltage_im[n,1,t]) - pyo.value(model.voltage_im[n,2,t])))*\
            (pyo.value(model.voltage_re[n,0,t]) - 0.5*(pyo.value(model.voltage_re[n,1,t]) + pyo.value(model.voltage_re[n,2,t])) + \
              math.sqrt(3)/2*(pyo.value(model.voltage_im[n,1,t]) - pyo.value(model.voltage_im[n,2,t]))) + \
            (pyo.value(model.voltage_im[n,0,t]) - math.sqrt(3)/2*(pyo.value(model.voltage_re[n,1,t]) - pyo.value(model.voltage_re[n,2,t])) - \
              0.5*(pyo.value(model.voltage_im[n,1,t]) + pyo.value(model.voltage_im[n,2,t])))*\
            (pyo.value(model.voltage_im[n,0,t]) - math.sqrt(3)/2*(pyo.value(model.voltage_re[n,1,t]) - pyo.value(model.voltage_re[n,2,t])) - \
              0.5*(pyo.value(model.voltage_im[n,1,t]) + pyo.value(model.voltage_im[n,2,t]))))/\
            ((pyo.value(model.voltage_re[n,0,t]) - 0.5*(pyo.value(model.voltage_re[n,1,t]) + pyo.value(model.voltage_re[n,2,t])) - \
              math.sqrt(3)/2*(pyo.value(model.voltage_im[n,1,t]) - pyo.value(model.voltage_im[n,2,t])))*\
            (pyo.value(model.voltage_re[n,0,t]) - 0.5*(pyo.value(model.voltage_re[n,1,t]) + pyo.value(model.voltage_re[n,2,t])) - \
              math.sqrt(3)/2*(pyo.value(model.voltage_im[n,1,t]) - pyo.value(model.voltage_im[n,2,t]))) + \
            (pyo.value(model.voltage_im[n,0,t]) + math.sqrt(3)/2*(pyo.value(model.voltage_re[n,1,t]) - pyo.value(model.voltage_re[n,2,t])) - \
              0.5*(pyo.value(model.voltage_im[n,1,t]) + pyo.value(model.voltage_im[n,2,t])))*\
            (pyo.value(model.voltage_im[n,0,t]) + math.sqrt(3)/2*(pyo.value(model.voltage_re[n,1,t]) - pyo.value(model.voltage_re[n,2,t])) - \
              0.5*(pyo.value(model.voltage_im[n,1,t]) + pyo.value(model.voltage_im[n,2,t])))))*100
        
        for i in model.p_gen:
                
            if i[0] in network.asymmetric_load.bus.values:
                if i[1] == 0:
                    p_gen_a_df.loc[i[2], i[0]] = pyo.value(model.p_gen[i])

                elif i[1] == 1:
                    p_gen_b_df.loc[i[2], i[0]] = pyo.value(model.p_gen[i])

                else:
                    p_gen_c_df.loc[i[2], i[0]] = pyo.value(model.p_gen[i])
    
    return v_a_df, v_b_df, v_c_df, vuf_df, p_gen_a_df, p_gen_b_df, p_gen_c_df
        
