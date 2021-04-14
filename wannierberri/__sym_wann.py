import numpy as np
import spglib
import numpy.linalg as la
from .atoms import Atoms
import sympy as sym
np.set_printoptions(precision=4,threshold=np.inf,linewidth=500)


class sym_wann():
    # could you break this line?
    # it is not obligatory to have every argument on a separate lines, 
    # but you could write it as groups of similar arguments per lines
    # also , it is a good idea to use spaces after commas .
    def __init__(self,HH_R=None,AA_R=None,BB_R=None,CC_R=None,SS_R=None,iRvec=None,nRvec=None,num_wann=None,spin=False,TR=True,lattice=None,symbols=None,positions=None,proj=None):
        self.HH_R = HH_R
        self.AA = False
        self.BB = False
        self.CC = False
        self.SS = False
        if AA_R is not None:
            self.AA=True
            self.AA_R=AA_R
        if BB_R is not None:
            self.BB=True
            self.BB_R=BB_R
        if CC_R is not None:
            self.CC=True
            self.CC_R=CC_R
        if SS_R is not None:
            self.SS=True
            self.SS_R=SS_R
        self.iRvec=iRvec.tolist()
        self.nRvec=nRvec
        self.num_wann=num_wann
        self.spin=spin
        self.TR=TR   # do you ever use Time-reversal ???
        self.lattice = lattice
        self.orbital_dic = {"s":1,"p":3,"d":5,"f":7,"sp3":4,"sp2":3,"l=0":1,"l=1":3,"l=2":5,"l=3":7}	
        projectiondic={}  #  alternatively : use defaultdic(lambda : "")  and the "if name in" statement is not needed
        self.wann_atom_info = []
        self.symbols_in=symbols
        self.positions_in=positions
        self.proj=proj
        num_wann_atom = 0
        self.num_atom = len(self.symbols_in)
        if self.spin: orb_spin = 2
        else: orb_spin = 1
        orbital_index_list=[]
        orb_op=[]
        orb_ed=[]
        for i in range(self.num_atom):
            orbital_index_list.append([])
            orb_op.append([])
            orb_ed.append([])
        orbital_index=0
        for npro in range(len(self.proj)):   #  for proj in self.proj :
            name = self.proj[npro].split(":")[0].split()[0] # proj.split(...
            orb = self.proj[npro].split(":")[1].strip('\n').split(';')
            if name in projectiondic.keys():    # or just " name in projectiondic "
                projectiondic[name]=projectiondic[name]+orb			
            else:
                newdic={name:orb}              #
                projectiondic.update(newdic)   # or just projectiondic[nema]=orb    
            for atom in range(self.num_atom):   # better to name the loop index variable starting with "i", e.g. iatom
                if self.symbols_in[atom] == name:
                    for orb_name in orb:
                        num_orb = orb_spin * self.orbital_dic[orb_name]
                        orbital_index_old = orbital_index
                        orbital_index+=num_orb
                        orb_op[atom] += [orbital_index_old]
                        orb_ed[atom] += [orbital_index]
                        orbital_index_list[atom] += [ i for i in range(orbital_index_old,orbital_index)]
        for atom in range(self.num_atom):
            name = self.symbols_in[atom]
            if name in projectiondic.keys():
                projection=projectiondic[name]
                num_wann_atom +=1
                self.wann_atom_info.append((atom+1,self.symbols_in[atom],self.positions_in[atom],projection,orbital_index_list[atom],orb_op[atom],orb_ed[atom]))
        self.num_wann_atom = num_wann_atom
        self.H_select=np.zeros((self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann),dtype=bool)  
        for atom_a in range(self.num_wann_atom):
            for atom_b in range(self.num_wann_atom):
                orb_name_a = self.wann_atom_info[atom_a][3]   # this is hard to follow 
                orb_name_b = self.wann_atom_info[atom_b][3]   # consider naking orb_name_a a liost of dictionaries 
                orb_op_a = self.wann_atom_info[atom_a][-2]    # so that each property has a readable name (key) 
                orb_op_b = self.wann_atom_info[atom_b][-2]
                orb_ed_a = self.wann_atom_info[atom_a][-1]
                orb_ed_b = self.wann_atom_info[atom_b][-1]
                for oa in range(len(orb_name_a)):    
                    for ob in range(len(orb_name_b)):
                        self.H_select[atom_a,atom_b,orb_op_a[oa]:orb_ed_a[oa],orb_op_b[ob]:orb_ed_b[ob]]=True
        for i in range(self.num_wann_atom):  # again : for info in self.wann_atom_info: print (info) 
            print(self.wann_atom_info[i])    # avoin using range to iterate over lists (unless necessay). also use enumerate and zip , if needed.

    def findsym(self):
        def show_symmetry(symmetry):
            for i in range(symmetry['rotations'].shape[0]):  #   for i,(rot,trans) in enumerate(zip( symmetry['rotations'] , symmetry['translations'] ) ): 
                print("  --------------- %4d ---------------" % (i + 1))
                rot = symmetry['rotations'][i]
                trans = symmetry['translations'][i]
                print("  rotation:")
                for x in rot:
                    print("     [%2d %2d %2d]" % (x[0], x[1], x[2]))
                print("  translation:")
                print("     (%8.5f %8.5f %8.5f)" % (trans[0], trans[1], trans[2]))
        atom_in=Atoms(symbols=self.symbols_in,cell=list(self.lattice),scaled_positions=self.positions_in,pbc=True)  
# is Atoms used just to be passed to spglib?
# spglib cal accept the "cell" as a tuple (lattice_vectors, reduced_atomic_positions, at_type_numbers)
# see irrep code 
        print("[get_spacegroup]")
        print("  Spacegroup of is %s." %spglib.get_spacegroup(atom_in))  # spacegroup of what? 
        self.symmetry = spglib.get_symmetry(atom_in)
        self.nsymm = self.symmetry['rotations'].shape[0]
        show_symmetry(self.symmetry)
        '''
        find a mathod can reduce operators to generators
        '''
    def get_angle(self,sina,cosa):
        if cosa > 1.0:
            cosa =1.0
        elif cosa < -1.0:
            cosa = -1.0
        alpha = np.arccos(cosa)
        if sina < 0.0:
            alpha = 2.0 * np.pi - alpha
        return alpha

    def rot_orb(self,orb_symbol,rot_glb):
        x = sym.Symbol('x')
        y = sym.Symbol('y')
        z = sym.Symbol('z')
        def ss(x,y,z): return 1+0*(x+y+z)    # optionally lambda-functions may be used here, like "ss = lambda x,y,z : 1+0*(x+y+z)
        def pz(x,y,z): return z
        def px(x,y,z): return x
        def py(x,y,z): return y
        def dz2(x,y,z): return (2*z*z-x*x-y*y)/(2*sym.sqrt(3.0))
        def dxz(x,y,z): return x*z
        def dyz(x,y,z): return y*z
        def dx2_y2(x,y,z): return (x*x-y*y)/2
        def dxy(x,y,z): return x*y
        def fz3(x,y,z): return z*(2*z*z-3*x*x-3*y*y)/(2*sym.sqrt(15.0))
        def fxz2(x,y,z): return x*(4*z*z-x*x-y*y)/(2*sym.sqrt(10.0))
        def fyz2(x,y,z): return y*(4*z*z-x*x-y*y)/(2*sym.sqrt(10.0))
        def fzx2_zy2(x,y,z): return z*(x*x-y*y)/2
        def fxyz(x,y,z): return x*y*z
        def fx3_3xy2(x,y,z): return x*(x*x-3*y*y)/(2*sym.sqrt(6.0))
        def f3yx2_y3(x,y,z): return y*(3*x*x-y*y)/(2*sym.sqrt(6.0))
        orb_s = [ss]
        orb_p = [pz,px,py]
        orb_d = [dz2,dxz,dyz,dx2_y2,dxy]
        orb_f = [fz3,fxz2,fyz2,fzx2_zy2,fxyz,fx3_3xy2,f3yx2_y3]
        orb_function_dic={'s':orb_s,'p':orb_p,'d':orb_d,'f':orb_f}
        orb_chara_dic={'s':[],'p':[z,x,y],'d':[z*z,x*z,y*z,x*x,x*y,y*y],'f':[z*z*z,x*z*z,y*z*z,z*x*x,x*y*z,x*x*x,y*y*y]}
        #  all the stuff above does not depend on input parameters, but it is initialized at every call of this method
        #  it is better to initialize it once and store in sopme structure ( class or dictionary )
        #  As I understand, hybrid orbitals are not implemented yes, but cam be easily added, just by writing linear combinations, right? 
        orb_dim = self.orbital_dic[orb_symbol]
        orb_rot_mat = np.zeros((orb_dim,orb_dim),dtype=float)
        xp = np.dot(np.linalg.inv(rot_glb)[0],np.transpose([x,y,z]))   #   could be written in one line ??
        yp = np.dot(np.linalg.inv(rot_glb)[1],np.transpose([x,y,z]))   #   xp,yp,zp= np.linalg.inv(rot_glb).dot(np.transpose([x,y,z]))
        zp = np.dot(np.linalg.inv(rot_glb)[2],np.transpose([x,y,z]))   # 
        rot_glb=np.array(list(rot_glb))
        OC = orb_chara_dic[orb_symbol]
        OC_len = len(OC)
        for i in range(orb_dim):
            e = (orb_function_dic[orb_symbol][i](xp,yp,zp)).expand()
            if orb_symbol == 's':
                orb_rot_mat[0,i] = e.subs(x,0).subs(y,0).subs(z,0).evalf()
            elif orb_symbol == 'p':
                for j in range(orb_dim):
                    etmp = e
                    orb_rot_mat[j,i] = etmp.subs(OC[j],1).subs(OC[(j+1)%OC_len],0).subs(OC[(j+2)%OC_len],0).evalf()
            elif orb_symbol == 'd':
                subs = []
                for j in range(orb_dim):
                    etmp = e
                    subs.append(etmp.subs(OC[j],1).subs(OC[(j+1)%OC_len],0).subs(OC[(j+2)%OC_len],0).subs(OC[(j+3)%OC_len],0).subs(OC[(j+4)%OC_len],0).subs(OC[(j+5)%OC_len],0))
                    if j == 0:
                        orb_rot_mat[j,i] = (subs[0]*sym.sqrt(3.0)).evalf()
                    elif j == 1:
                        orb_rot_mat[j,i] = subs[1].evalf()
                    elif j == 2:
                        orb_rot_mat[j,i] = subs[2].evalf()
                    elif j == 3:
                        orb_rot_mat[j,i] = (2*subs[3]+subs[0]).evalf()
                    elif j == 4:
                        orb_rot_mat[j,i] = subs[4].evalf()
            elif orb_symbol == 'f':
                subs = []
                for j in range(orb_dim):		
                    etmp = e
                    subs.append(etmp.subs(OC[j],1).subs(OC[(j+1)%OC_len],0).subs(OC[(j+2)%OC_len],0).subs(OC[(j+3)%OC_len],0).subs(OC[(j+4)%OC_len],0).subs(OC[(j+5)%OC_len],0).subs(OC[(j+6)%OC_len],0))
                    if j == 0:
                        orb_rot_mat[0,i] = (subs[0]*sym.sqrt(15.0)).evalf()
                    if j == 1:  # why not "elif" ?
                        orb_rot_mat[1,i] = (subs[1]*sym.sqrt(10.0)/2).evalf()
                    if j == 2:
                        orb_rot_mat[2,i] = (subs[2]*sym.sqrt(10.0)/2).evalf()
                    if j == 3:
                        orb_rot_mat[3,i] = (2*subs[3]+3*subs[0]).evalf()
                    if j == 4:
                        orb_rot_mat[4,i] = subs[4].evalf()
                    if j == 5:
                        orb_rot_mat[5,i] = ((2*subs[5]+subs[1]/2)*sym.sqrt(6.0)).evalf()
                    if j == 6:
                        orb_rot_mat[6,i] = ((-2*subs[6]-subs[2]/2)*sym.sqrt(6.0)).evalf()
        return orb_rot_mat
	
    def Part_P(self,rot_sym_glb,orb_symbol):
        if abs(np.dot(np.transpose(rot_sym_glb),rot_sym_glb) - np.eye(3)).sum() >1.0E-4:
            print('rot_sym is not orthogomal \n {}'.format(rot_sym_glb))
        rmat = np.linalg.det(rot_sym_glb)*rot_sym_glb
        #rmat = rot_sym_glb
        select = np.abs(rmat) < 0.01
        rmat[select] = 0.0 
        select = rmat > 0.99 
        rmat[select] = 1.0 
        select = rmat < -0.99 
        rmat[select] = -1.0 
        if self.spin:
            if np.abs(rmat[2,2]) < 1.0:
                print('one')	
                beta = np.arccos(rmat[2,2])
                cos_gamma = -rmat[2,0] / np.sin(beta)
                sin_gamma =  rmat[2,1] / np.sin(beta)
                gamma = self.get_angle(sin_gamma, cos_gamma)
                cos_alpha = rmat[0,2] / np.sin(beta)
                sin_alpha = rmat[1,2] / np.sin(beta)
                alpha = self.get_angle(sin_alpha, cos_alpha)
            else:
                beta = 0.0
                if rmat[2,2] == -1. :beta = np.pi
                gamma = 0.0
                alpha = np.arccos(rmat[1,1])
                if rmat[0,1] > 0.0:alpha = -1.0*alpha
            euler_angle = np.array([alpha,beta,gamma])
            dmat = np.zeros((2,2),dtype=complex)
            dmat[0,0] =  np.exp(-(alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
            dmat[0,1] = -np.exp(-(alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
            dmat[1,0] =  np.exp( (alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0)
            dmat[1,1] =  np.exp( (alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0)
            # an alternative way is here https://github.com/stepan-tsirkin/irrep/blob/c569ab17ec1f5c016340acacd4004742dc28e8b8/irrep/spacegroup.py#L52-L53
            # one needs only the rotational axis and the  angle, the latter can have only few discrete values.
        rot_orbital = self.rot_orb(orb_symbol,rot_sym_glb) # can this line be moved above "if self.spin:"
        if self.spin:
            rot_orbital = np.kron(rot_orbital,dmat)
            rot_imag = rot_orbital.imag
            rot_real = rot_orbital.real
            rot_imag[abs(rot_imag) < 10e-6] = 0  # Why is this clean-up necessary ? 
            rot_real[abs(rot_real) < 10e-6] = 0  # 
            rot_orbital = np.array(rot_real + 1j*rot_imag,dtype=complex)
        return rot_orbital

    def atom_rot_map(self,sym):
        wann_atom_positions = [self.wann_atom_info[i][2] for i in range(self.num_wann_atom)]
        rot_map=[]
        vec_shift_map=[]
        for atomran in range(self.num_wann_atom):
            atom_position=np.array(wann_atom_positions[atomran])
            new_atom =np.round( np.dot(self.symmetry['rotations'][sym],atom_position) + self.symmetry['translations'][sym],decimals=5)
            for atom_index in range(self.num_wann_atom):
                old_atom= np.round(np.array(wann_atom_positions[atom_index]),decimals=5)
                diff = np.array(np.round(new_atom-old_atom,decimals=8))
                if abs(diff[0]%1)+abs(diff[1]%1)+abs(diff[2]%1)<10E-5:
                    match_index=atom_index
                    vec_shift= np.array(np.round(new_atom-np.array(wann_atom_positions[match_index]),decimals=2),dtype=int)
                else:
                    if atom_index==self.num_wann_atom-1:
                        assert atom_index != 0,'Error!!!!: no atom can match the new one Rvec = {}, atom_index = {}'.format(self.iRvec[ir],atom_index)
            rot_map.append(match_index)
            vec_shift_map.append(vec_shift)
        return np.array(rot_map,dtype=int),np.array(vec_shift_map,dtype=int)

    def full_p_mat(self,atom_index,rot):
        orbitals = self.wann_atom_info[atom_index][3]
        op = self.wann_atom_info[atom_index][-2]
        ed = self.wann_atom_info[atom_index][-1]
        p_mat = np.zeros((self.num_wann,self.num_wann),dtype = complex)
        p_mat_dagger = np.zeros((self.num_wann,self.num_wann),dtype = complex)
        rot_sym = self.symmetry['rotations'][rot]
        rot_sym_glb = np.dot(np.dot(np.transpose(self.lattice),rot_sym),np.linalg.inv(np.transpose(self.lattice)) )
        for orb in range(len(orbitals)):
            orb_name = orbitals[orb]
            tmp = self.Part_P(rot_sym_glb,orb_name)
            p_mat[op[orb]:ed[orb],op[orb]:ed[orb]] = tmp 	
            p_mat_dagger[op[orb]:ed[orb],op[orb]:ed[orb]] = np.conj(np.transpose(tmp)) 	
        return p_mat,p_mat_dagger

    def Mat_trans(self,HH_R): 
            HH_R_spin = HH_R*0.0
            HH_R_spin[0:self.num_wann:2,0:self.num_wann:2,:] = HH_R[0:self.num_wann//2,0:self.num_wann//2,:]
            HH_R_spin[1:self.num_wann:2,0:self.num_wann:2,:] = HH_R[self.num_wann//2:self.num_wann,0:self.num_wann//2,:]
            HH_R_spin[0:self.num_wann:2,1:self.num_wann:2,:] = HH_R[0:self.num_wann//2,self.num_wann//2:self.num_wann,:]
            HH_R_spin[1:self.num_wann:2,1:self.num_wann:2,:] = HH_R[self.num_wann//2:self.num_wann,self.num_wann//2:self.num_wann,:]
            return HH_R_spin*1.0	
    
    def Mat_trans_back(self,HH_R):
            HH_R_re = HH_R*0.0
            HH_R_re[0:self.num_wann//2,0:self.num_wann//2,:]                     = HH_R[0:self.num_wann:2,0:self.num_wann:2,:]
            HH_R_re[self.num_wann//2:self.num_wann,0:self.num_wann//2,:]         = HH_R[1:self.num_wann:2,0:self.num_wann:2,:]
            HH_R_re[0:self.num_wann//2,self.num_wann//2:self.num_wann,:]         = HH_R[0:self.num_wann:2,1:self.num_wann:2,:]
            HH_R_re[self.num_wann//2:self.num_wann,self.num_wann//2:self.num_wann,:] = HH_R[1:self.num_wann:2,1:self.num_wann:2,:]
            return  HH_R_re*1.0
    
    def symmetrize(self):
        self.findsym()
        if self.spin:
            self.HH_R = self.Mat_trans(self.HH_R)
            
            if self.AA: self.AA_R= self.Mat_trans(self.AA_R)
            if self.BB: self.BB_R= self.Mat_trans(self.BB_R)
            if self.CC: self.CC_R= self.Mat_trans(self.CC_R)
            if self.SS: self.SS_R= self.Mat_trans(self.SS_R)
        
    # what is the point of having this fubction insiude the "symmetrize" ?
    # anyway, it is a method of the class, and it can modify the properies of the object 

        def average_H(self,H_res,iRvec,keep_New_R=True):
            R_list = np.array(iRvec,dtype=int)
            nRvec=len(R_list)
            tmp_R_list = []
            shape = np.shape(H_res)
            if self.AA: A_res=np.zeros((shape[0],shape[1],shape[2],3),dtype=complex)
            else:A_res=0.0; self.AA_R=None
            if self.BB: B_res=np.zeros((shape[0],shape[1],shape[2],3),dtype=complex)
            else:B_res=0.0; self.BB_R=None
            if self.CC: C_res=np.zeros((shape[0],shape[1],shape[2],3),dtype=complex)
            else:C_res=0.0; self.CC_R=None
            if self.SS: S_res=np.zeros((shape[0],shape[1],shape[2],3),dtype=complex)
            else:S_res=0.0; self.SS_R=None
            for rot in range(0,self.nsymm):
                print('Sym = ',rot)
                p_map = np.zeros((self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                p_map_dagger = np.zeros((self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                for atom in range(self.num_wann_atom):
                    p_map[atom],p_map_dagger[atom] = self.full_p_mat(atom,rot)
                rot_map,vec_shift = self.atom_rot_map(rot)
                R_map = np.dot(R_list,np.transpose(self.symmetry['rotations'][rot]))
                atom_R_map = R_map[:,None,None,:] - vec_shift[None,:,None,:] + vec_shift[None,None,:,:]
                HH_all = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann),dtype=complex)
                if self.AA: AA_all = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann,3),dtype=complex)
                if self.BB: BB_all = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann,3),dtype=complex)
                if self.CC: CC_all = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann,3),dtype=complex)
                if self.SS: SS_all = np.zeros((nRvec,self.num_wann_atom,self.num_wann_atom,self.num_wann,self.num_wann,3),dtype=complex)
                for iR in range(nRvec):
                        for atom_a in range(self.num_wann_atom):
                            for atom_b in range(self.num_wann_atom):
                                new_Rvec=list(atom_R_map[iR,atom_a,atom_b])
                                if new_Rvec in self.iRvec:
                                    new_Rvec_index = self.iRvec.index(new_Rvec)
                                    HH_all[iR,atom_a,atom_b,self.H_select[atom_a,atom_b]] = self.HH_R[self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index]
                                    if self.AA: AA_all[iR,atom_a,atom_b,self.H_select[atom_a,atom_b],:] = self.AA_R[self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index,:]
                                    if self.BB: BB_all[iR,atom_a,atom_b,self.H_select[atom_a,atom_b],:] = self.BB_R[self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index,:]
                                    if self.CC: CC_all[iR,atom_a,atom_b,self.H_select[atom_a,atom_b],:] = self.CC_R[self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index,:]
                                    if self.SS: SS_all[iR,atom_a,atom_b,self.H_select[atom_a,atom_b],:] = self.SS_R[self.H_select[rot_map[atom_a],rot_map[atom_b]],new_Rvec_index,:]
                                else: 
                                    if new_Rvec in tmp_R_list:
                                        bear=1    # what is this for ?
                                    else:
                                        tmp_R_list.append(new_Rvec)		
                for atom_a in range(self.num_wann_atom):
                        for atom_b in range(self.num_wann_atom):
                            HH_tmp = np.dot(np.dot(p_map_dagger[atom_a],HH_all[:,atom_a,atom_b]),p_map[atom_b])
                            H_res += HH_tmp.transpose(0,2,1)
                            for i in range(3):
                                if self.AA:
                                    AA_tmp = np.dot(np.dot(p_map_dagger[atom_a],AA_all[:,atom_a,atom_b,:,:,i]),p_map[atom_b])
                                    A_res[:,:,:,i] += AA_tmp.transpose(0,2,1)
                                if self.BB: 
                                    BB_tmp = np.dot(np.dot(p_map_dagger[atom_a],BB_all[:,atom_a,atom_b,:,:,i]),p_map[atom_b])
                                    B_res[:,:,:,i] += BB_tmp.transpose(0,2,1)
                                if self.CC:
                                    CC_tmp = np.dot(np.dot(p_map_dagger[atom_a],CC_all[:,atom_a,atom_b,:,:,i]),p_map[atom_b])
                                    C_res[:,:,:,i] += CC_tmp.transpose(0,2,1)
                                if self.SS:
                                    SS_tmp = np.dot(np.dot(p_map_dagger[atom_a],SS_all[:,atom_a,atom_b,:,:,i]),p_map[atom_b])
                                    S_res[:,:,:,i] += SS_tmp.transpose(0,2,1)
                           #  here you need to rotatye the vectors (with rotatin matrices in cartesian coordinate R) , 
                           #  and to pseudo-vectors multiply by det (R) which is +1 for pure rotations and -1 for Inversion and mirrors
            
            if keep_New_R:
                return  {'HH':H_res/self.nsymm,'AA':A_res/self.nsymm,'BB':B_res/self.nsymm,'CC':C_res/self.nsymm,'SS':S_res/self.nsymm},tmp_R_list
            else:
                return  {'HH':H_res/self.nsymm,'AA':A_res/self.nsymm,'BB':B_res/self.nsymm,'CC':C_res/self.nsymm,'SS':S_res/self.nsymm}
        H_res_exist = self.HH_R*0.0
        XX_R_re1, iRvec_add =  average_H(self,H_res_exist,self.iRvec,keep_New_R=True)
        nRvec_add = len(iRvec_add)
        H_res_add=np.zeros((self.num_wann,self.num_wann,nRvec_add),dtype=complex)
        XX_R_re2  =  average_H(self,H_res_add,iRvec_add,keep_New_R=False)
        HH_R_a = np.zeros((self.num_wann,self.num_wann,self.nRvec+nRvec_add),dtype=complex)
        HH_R_a[:,:,:self.nRvec]=XX_R_re1['HH']
        HH_R_a[:,:,self.nRvec:]=XX_R_re2['HH']
        if self.AA:
            AA_R_a = np.zeros((self.num_wann,self.num_wann,self.nRvec+nRvec_add,3),dtype=complex)
            AA_R_a[:,:,:self.nRvec]=XX_R_re1['AA']
            AA_R_a[:,:,self.nRvec:]=XX_R_re2['AA']
            if self.spin:
                self.AA_R = self.Mat_trans_back(AA_R_a)
            else: self.AA_R = AA_R_a*1.0
        if self.BB: 
            BB_R_a = np.zeros((self.num_wann,self.num_wann,self.nRvec+nRvec_add,3),dtype=complex)
            BB_R_a[:,:,:self.nRvec]=XX_R_re1['BB']
            BB_R_a[:,:,self.nRvec:]=XX_R_re2['BB']
            if self.spin:
                self.BB_R = self.Mat_trans_back(BB_R_a)
            else: self.BB_R = BB_R_a*1.0
        if self.CC: 
            CC_R_a = np.zeros((self.num_wann,self.num_wann,self.nRvec+nRvec_add,3),dtype=complex)
            CC_R_a[:,:,:self.nRvec]=XX_R_re1['CC']
            CC_R_a[:,:,self.nRvec:]=XX_R_re2['CC']
            if self.spin:
                self.CC_R = self.Mat_trans_back(CC_R_a)
            else: self.CC_R = CC_R_a*1.0
        if self.SS: 
            SS_R_a = np.zeros((self.num_wann,self.num_wann,self.nRvec+nRvec_add,3),dtype=complex)
            SS_R_a[:,:,:self.nRvec]=XX_R_re1['SS']
            SS_R_a[:,:,self.nRvec:]=XX_R_re2['SS']
            if self.spin:   
                self.SS_R = self.Mat_trans_back(SS_R_a)
            else: self.SS_R = SS_R_a*1.0  # can we have the SS matrix in a scalar calculation ?
        if self.spin:
            self.HH_R = self.Mat_trans_back(HH_R_a)
        else:
            self.HH_R = HH_R_a*1.0
        self.nRvec += nRvec_add
        self.iRvec += iRvec_add
        
        return {'HH_R':self.HH_R,'AA_R':self.AA_R,'BB_R':self.BB_R,'CC_R':self.CC_R,'SS_R':self.SS_R,'nRvec':self.nRvec,'iRvec':self.iRvec}

### 
### Generally, it is better to separate the code into two parts :
### 1st you construct the transformation "scheme", i.e. :
### which atom transforms into which, with what rotation matrices (p_map , p_map_dagger)
### to which iRvec it transforme, what are new iRvec_add, etc. 
### Then you write a function that actually symmetyryzes smth which may be 
#      - scalar (Hamiltonian)
#      - vector (AA,BB,CC,SS ....)
#      - tensor ... (not needed so far, but in future may be easiuly generalized.
#      for vectors, you just multiply them by rotation matrices (in the cartesian coordinates)
#      the vector may be inversion-odd (true vector AA,BB) or inversion-even (pseudovector - SS, CC)
#      (also TR-odd or TR-even, but you do not use TR so far)
#       for pseudoverctor you just multiply it again by det(R) , which is -1 for operations involving inversion (Inversion and Mirrirs)
#
# thus the code will be shorter and clearer.





