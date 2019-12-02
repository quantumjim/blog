from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info.synthesis import euler_angles_1q

from pairwise_tomography.pairwise_state_tomography_circuits import pairwise_state_tomography_circuits
from pairwise_tomography.pairwise_fitter import PairwiseStateTomographyFitter
from pairwise_tomography.utils import concurrence
from pairwise_tomography.visualization import draw_entanglement_graph

from numpy import pi, cos, sin, sqrt, exp, arccos, arctan2, conj, array


class ParameterGraph ():
    
    def __init__ (self,num_qubits,coupling_map=[],labels=None,variables={}):
        
        self.num_qubits = num_qubits
        
        self.expect = {}
        for j in range(self.num_qubits):
            self.expect[j] = {'X':0.0,'Y':0.0,'Z':1.0}
        
        self.coupling_map = []
        for j in range(self.num_qubits-1):
            for k in range(j+1,self.num_qubits):
                if ([j,k] in coupling_map) or ([j,k] in coupling_map) or (not coupling_map):
                    self.coupling_map.append([j,k])
                    self.expect[j,k] = {'XX':0.0,'XY':0.0,'XZ':0.0,
                                        'YX':0.0,'YY':0.0,'YZ':0.0,
                                        'ZX':0.0,'ZY':0.0,'ZZ':1.0}
        
        if not labels:
            self.labels = ['Qubit '+str(j) in range(self.num_qubits)]
        else:
            self.labels = labels
                    
        self.variables = variables
        self.paulis_used = []
        for pauli in ['X','Y','Z','XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
            if pauli in self.variables:
                self.paulis_used.append(pauli)
            else:
                self.variables[pauli] = pauli
                    
        self.qc = QuantumCircuit(self.num_qubits)
        
    def _update_expect(self):
        
        tomo_circs = pairwise_state_tomography_circuits(self.qc, self.qc.qregs[0])
        job = execute(tomo_circs, Aer.get_backend('qasm_simulator'), shots=8192)
        raw_expect = PairwiseStateTomographyFitter(job.result(), tomo_circs, self.qc.qregs[0]).fit(output='expectation')
        
        self.expect = {}
        for j in range(self.num_qubits):
            self.expect[j] = {'X':0.0,'Y':0.0,'Z':0.0}
        degree = {j:0 for j in range(self.num_qubits)}
        for pair in self.coupling_map:
            (x,y) = tuple(pair)
            self.expect[x,y] = {}
            for paulis in raw_expect[x,y]:
                if 'I' not in paulis:
                    self.expect[x,y][paulis[0]+paulis[1]] = raw_expect[x,y][paulis]
                else:
                    pauli = list(paulis)
                    pauli.remove('I')
                    pauli = pauli[0]
                    k = paulis.index(pauli)
                    self.expect[pair[k]][pauli] += raw_expect[tuple(pair)][paulis]
                    degree[pair[k]] += 1/3
        for j in range(self.num_qubits):
            for pauli in ['X','Y','Z']:
                self.expect[j][pauli] = self.expect[j][pauli]/degree[j]
    
    def get_state(self,qubit_label,verbose=False):
        
        qubit = self.labels.index(qubit_label)
        state = {}
        for pauli in ['X','Y','Z']:
            if (pauli in self.paulis_used):
                state[self.variables[pauli]] = self.expect[qubit][pauli]
            elif verbose:
                state[pauli] = self.expect[qubit][pauli]
        return state
    
    def get_relationship(self,qubit_label_1,qubit_label_2,verbose=False):  
        qubit1 = self.labels.index(qubit_label_1)
        qubit2 = self.labels.index(qubit_label_2)
        q1,q2 = sorted([qubit1,qubit2])        
        qubit = (q1,q2)
        relationship = {}
        for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
            if (pauli in self.paulis_used):
                if type(self.variables[pauli])==tuple:
                    variable = self.variables[pauli][0] +' '+ self.labels[q1] +' '+ self.variables[pauli][1] +' '+ self.labels[q2]
                else:
                    variable = self.variables[pauli]
                relationship[variable] = self.expect[qubit][pauli]
            elif verbose:
                relationship[pauli] = self.expect[qubit][pauli]
        return relationship
    
    def set_state(self,target_variables,qubit_label,q_if=None):
        
        def basis_change(pole,basis,qubit,dagger=False):
            '''
                Returns the circuit required to change from the Z basis to the eigenbasis
                of a particular Pauli. The opposite is done when `dagger=True`.
            '''
            
            if pole=='+' and dagger==True:
                self.qc.x(qubit)
            
            if basis=='X':
                self.qc.h(qubit)
            elif basis=='Y':
                if dagger:
                    self.qc.rx(-pi/2,qubit)
                else:
                    self.qc.rx(pi/2,qubit)
                    
            if pole=='+' and dagger==False:
                self.qc.x(qubit)
                    
        
        def normalize(expect):
            
            for pauli in ['X','Y','Z']:
                if pauli not in expect:
                    expect[pauli] = self.expect[qubit][pauli]
            
            R = sqrt( expect['X']**2 + expect['Y']**2 + expect['Z']**2 )
            
            return {pauli:expect[pauli]/R for pauli in expect}
        
        def get_basis(expect):
            
            normalized_expect = normalize(expect)
            
            theta = arccos(normalized_expect['Z'])
            phi = arctan2(normalized_expect['Y'],normalized_expect['X'])
            
            state0 = [cos(theta/2),exp(1j*phi)*sin(theta/2)]
            state1 = [conj(state0[1]),-conj(state0[0])]
            
            return [state0,state1]

        target_expect = {}
        for label in target_variables:
            pauli = list(self.variables.keys())[list(self.variables.values()).index(label)]
            target_expect[pauli] = target_variables[label]
        
        qubit = self.labels.index(qubit_label)
        
        current_basis = get_basis(self.get_state(self.labels[qubit]))
        target_basis = get_basis(target_expect)
        
        U = array([ [0 for _ in range(2)] for _ in range(2) ], dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[j][k] += target_basis[i][j]*conj(current_basis[i][k])
                
        the,phi,lam = euler_angles_1q(U)
        
        if q_if:
            control_variable,pole,control_qubit_label = q_if[0],q_if[1],q_if[2]
            control_qubit = self.labels.index(control_qubit_label)
            control_pauli = list(self.variables.keys())[list(self.variables.values()).index(control_variable)]
            basis_change(pole,control_pauli,control_qubit,dagger=False)
            self.qc.cu3(the,phi,lam,control_qubit,qubit)
            basis_change(pole,control_pauli,control_qubit,dagger=True)
        else:
            self.qc.u3(the,phi,lam,qubit)
                
        self._update_expect()
        
              
    def swap (self,qubit_label_1,qubit_label_2,fraction):
        ''''''
        
        qubit1 = self.labels.index(qubit_label_1)
        qubit2 = self.labels.index(qubit_label_2)
        
        self.qc.cx(qubit1,qubit2)
        #self.qc.cy(qubit2,qubit1)
        self.qc.mcrx(pi*fraction,[self.qc.qregs[0][qubit2]],self.qc.qregs[0][qubit1])
        self.qc.cx(qubit1,qubit2)
        
        self._update_expect()
        
    def print_states(self,dp=3):
        
        for j in range(self.num_qubits):
            print('\nState of qubit',j)
            for pauli in ['X','Y','Z']:
                print('   ',pauli+':',round(self.expect[j][pauli],dp))

        for pair in self.coupling_map:
            (j,k) = tuple(pair)
            if (j,k) in self.expect:
                print('\nRelationship for qubits',j,'and',k)
                for paulis in self.expect[j,k]:
                    print('   ',paulis+':',round(self.expect[j,k][paulis],dp))