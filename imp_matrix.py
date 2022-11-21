import pandapower as pp
import numpy as np
import cmath

def impedance_matrix(network):
    
    pp.runpp_3ph(network)
    
    z_0 = network._ppc0['branch']
    z_1 = network._ppc1['branch']
    z_2 = network._ppc2['branch']
    
    z_012 = {}
    for i in range(0, (len(network.line.index) + len(network.trafo.index))):
        z_012[int(np.real(z_0[i][0])), int(np.real(z_0[i][1]))] = np.zeros([3,3], dtype = complex)
        
        z_012[int(np.real(z_0[i][0])), int(np.real(z_0[i][1]))][0,0] = complex(np.real(z_0[i][2]), \
                                                                                np.real(z_0[i][3]))
        z_012[int(np.real(z_0[i][0])), int(np.real(z_0[i][1]))][1,1] = complex(np.real(z_1[i][2]), \
                                                                                np.real(z_1[i][3]))
        z_012[int(np.real(z_0[i][0])), int(np.real(z_0[i][1]))][2,2] = complex(np.real(z_2[i][2]), \
                                                                                np.real(z_2[i][3]))
    
    
    # Used for transformation between sequence and phase systems
    a1 = cmath.rect(1, 2/3*cmath.pi)
    a2 = cmath.rect(1, 4/3*cmath.pi)
    
    matrix_A = np.matrix([[1, 1, 1], [1, a2, a1], [1, a1, a2]])
    
    z_abc = {}
    r_abc = {}
    x_abc = {}
    
    for i in z_012:
        z_abc[i] = np.matmul(np.matmul(matrix_A, z_012[i]), np.linalg.inv(matrix_A))
        r_abc[i] = np.real(z_abc[i])
        x_abc[i] = np.imag(z_abc[i])
    
    #From and to nodes are sorted from 0 to n - total number of nodes
    #Nodes can be defined with unsroted indices, e.g., 1, 6, 13, 27, etc.
    #Following operation will match the sorted values of from and to nodes with ones defined with the network creation
    
    nodes_unsrt = network.bus.index.values.tolist()
    nodes_unsrt.sort()
    
    nodes_srt = []
    
    ft_nodes_unsrt = []
    to_nodes_unsrt = []
    
    for i in z_abc.keys():
        
        ft_nodes_unsrt.append((i[0], i[1]))
             
        if i[0] not in nodes_srt:
            nodes_srt.append(i[0])
        if i[1] not in nodes_srt:
            nodes_srt.append(i[1])
    
    nodes_srt.sort()
    ft_nodes_unsrt.sort()
    
    ft_nodes_srt = ft_nodes_unsrt.copy()

    
    for i in range(0, len(ft_nodes_unsrt)):
        for j in range(0, len(nodes_srt)):
            if ft_nodes_unsrt[i][0] == nodes_srt[j] and nodes_srt[j] != nodes_unsrt[j]:
                ft_nodes_unsrt_list = list(ft_nodes_unsrt[i])
                ft_nodes_unsrt_list[0] = nodes_unsrt[j]
                ft_nodes_unsrt[i] = tuple(ft_nodes_unsrt_list)
                break
            if ft_nodes_unsrt[i][1] == nodes_srt[j] and nodes_srt[j] != nodes_unsrt[j]:
                ft_nodes_unsrt_list = list(ft_nodes_unsrt[i])
                ft_nodes_unsrt_list[1] = nodes_unsrt[j]
                ft_nodes_unsrt[i] = tuple(ft_nodes_unsrt_list)
                break
    
    z_abc_unsrt = {}
    r_abc_unsrt = {}
    x_abc_unsrt = {}
    
    for i in range(0, len(r_abc)):
        z_abc_unsrt[ft_nodes_unsrt[i]] = z_abc[ft_nodes_srt[i]]
        r_abc_unsrt[ft_nodes_unsrt[i]] = r_abc[ft_nodes_srt[i]]
        x_abc_unsrt[ft_nodes_unsrt[i]] = x_abc[ft_nodes_srt[i]]
        
    y_abc_unsrt = {}    
    g_abc_unsrt = {} 
    b_abc_unsrt = {} 
    
    for i in z_abc_unsrt:
        y_abc_unsrt[i] = np.linalg.inv(z_abc_unsrt[i])
        g_abc_unsrt[i] = np.real(y_abc_unsrt[i])
        b_abc_unsrt[i] = np.imag(y_abc_unsrt[i])
    
    return r_abc_unsrt, x_abc_unsrt, g_abc_unsrt, b_abc_unsrt


        
    