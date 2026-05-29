########################
# 07/03/2019 A. Alekseenko
# This module contains functions and
# subroutines that read and write data to hard drive
########################

##########################
# my_get_file_names
# This subroutine returns a list
# of filenames that are located in
# directory given by path. Files will be
# located in subdirectories and listed with
# path, starting with the provided path.
#
###########################
def my_get_file_names (path):
    import os

    files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    return files


#########################
# my_get_file_names
# This subroutine returns a list
# of filenames that are located in
# directory given by path. Files will be
# located in subdirectories and listed with
# path, starting with the provided path.
#
###########################
def my_get_soltn_file_names (path):
    import os

    files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '_SltnColl' in file:
                files.append(os.path.join(r, file))

    return files

#########################
# my_get_file_names_time(path, cutoff_time)
#
# This is a modification of the above subroutine. It only appends names
# of saved files to the list if the time stamp is before the provided time valur.
#
# This subroutine returns a list
# of filenames that are located in
# directory given by path. Files will be
# located in subdirectories and listed with
# path, starting with the provided path.
#
###########################
def my_get_soltn_file_names_time (path, cutoff_time):
    import os

    files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '_SltnColl' in file:
                time_stamp=extract_time_name(file)
                if time_stamp <= cutoff_time:
                    files.append(os.path.join(r, file))

    return files


############################
# my_read_solution (path)
#
# This subroutine reads a solution from the
# saved DGV0D3V *_SltnColl.dat file and returns an numpy array
#
# variable path has absolute path to the file to read including the file name
##############################
def my_read_solution (path):
    #open binary file
    savefile=open(path, 'rb')

    #the first write is to record time to which this save corresponds.
    # the time variable is double precision and each write in fortran ads 32 bits before and after the written staff.
    # so 4 bytes+8 bytes+ 4bytes is 16
    bytes=savefile.read(16)
    # now that we got the first piece of information, let us unpack it using struct
    import struct
    data=struct.unpack('=ldl',bytes)
    scr1=data[0]
    t=data[1]
    scr2=data[2]
    #the second piece of information is the length of the solution array and the collision operator array,
    #not clear what the size of the array is, will try for long integer (4 bytes)
    # 4 bytes padding, 4 bytes the files size and 4 bytes
    bytes=savefile.read(12)
    data=struct.unpack('=lll',bytes)
    scr1 = data[0]    #Fortran writes the total length of data in bytes in a write statements in the beginning and in the end of the record
    fsize = data[1]
    scr2 = data[2]
    #next we need to read fsize double reals into a sequence (not sure what it will be, list or tuple
    bytes1 = savefile.read(4)
    two_arrays_len=struct.unpack('=l',bytes1)  # this should be the length of solution and collision operator arrays
    bytes = savefile.read(fsize*8)
    fmt='='+str(fsize)+"d"
    solscr=struct.unpack(fmt,bytes)
    # next we convert the sequence into a numpy array to return it.
    import numpy
    sol=numpy.array(solscr).reshape(1,fsize)
    # debug
    #sh=sol.shape
    # end debug
    # close the file
    savefile.close()
    return sol, fsize

############################
# my_read_solution_trim (path,M,Mtrim)
#
# This subroutine reads a solution from the
# saved DGV0D3V *_SltnColl.dat file. It trims boundary cells and returns
# an array on a submehs inside of the domain. The submesh is formed by
# removing Mtrim cells on each end of the 1D velocity interval for ecah velocity variable
#
# we expect that the velocity mesh has the same number of cells in each dimension MxMxM
# Variables:
# path === pfull ath to solution file including the solution file name
# M -- size of the mesh used for the discrete solution
# Mtrim -- # cells to trim on each end of the 1D interval
#
#
##############################
def my_read_solution_trim (path, M, Mtrim):
    #open binary file
    savefile=open(path, 'rb')

    #the first write is to record time to which this save corresponds.
    # the time variable is double precision and each write in fortran ads 32 bits before and after the written staff.
    # so 4 bytes+8 bytes+ 4bytes is 16
    bytes=savefile.read(16)
    # now that we got the first piece of information, let us unpack it using struct
    import struct
    data=struct.unpack('=ldl',bytes)
    scr1=data[0]
    t=data[1]
    scr2=data[2]
    #the second piece of information is the length of the solution array and the collision operator array,
    #not clear what the size of the array is, will try for long integer (4 bytes)
    # 4 bytes padding, 4 bytes the files size and 4 bytes
    bytes=savefile.read(12)
    data=struct.unpack('=lll',bytes)
    scr1 = data[0]    #Fortran writes the total length of data in bytes in a write statements in the beginning and in the end of the record
    fsize = data[1]
    scr2 = data[2]
    #next we need to read fsize double reals into a sequence (not sure what it will be, list or tuple
    bytes1 = savefile.read(4)
    two_arrays_len=struct.unpack('=l',bytes1)  # this should be the length of solution and collision operator arrays
    bytes = savefile.read(fsize*8)
    fmt='='+str(fsize)+"d"
    solscr=struct.unpack(fmt,bytes)
    # next we convert the sequence into a numpy array to return it.
    import numpy
    sol=numpy.array(solscr).reshape(1,fsize)
    # debug
    #sh=sol.shape
    # end debug
    # close the file
    savefile.close()
    # The solution has been read from file into an array. Now we will need to trim it.
    # a quick check that the supplied values are consistens:
    if M**3 != fsize:
       print('my_read_solution_trim: the value M is not consistent with the solution size. Stop')
       exit()
    if 2*Mtrim > M:
       print('my_read_solution_trim: the provided value Mtrip is too large for the number of cells M. Stop')
       exit()
    ###################################
    # We are ready to trim the array/
    Mnew=M-2*Mtrim
    nsze=Mnew**3
    soltrm = numpy.zeros((1,nsze))
    for i in range(Mnew):
        for j in range(Mnew):
            for k in range(Mnew):
                soltrm[0,i*Mnew*Mnew+j*Mnew+k]=sol[0,(Mtrim+i)*M*M+(Mtrim+j)*M+Mtrim+k ]
    # the trimmed solution has been recorded.
    # Next we will check magnitudes of the truncated values and issue a warning if
    # Large values were truncated
    treshld=1.0e-2
    lrg_val_flg=False
    for i in range(fsize):
      # unwrap the index
      iu = i // (M*M)
      iv = (i-iu*M*M) // (M)
      iw =  i-iu*M*M-iv*M
      if (iu<Mtrim) or (iu > (M - 1 - Mtrim)) or (iv < Mtrim) or (iv > (M - 1 - Mtrim)) or (iw < Mtrim) or (iw > (M - 1 - Mtrim)):
          if numpy.abs(sol[0,i]) >= treshld:
               lrg_val_flg=True
    if lrg_val_flg:
      print('my_read_solution_trim: Attention truncating at least one large value. treshld=', treshld)
    return soltrm, nsze


###########################################################
## solution_trim
# We are ready to trim the array/

def solution_trim(sol,M,Mtrim):
    import numpy
    Mnew=M-2*Mtrim
    nsze=Mnew**3
    soltrm = numpy.zeros((1,nsze))
    for i in range(Mnew):
        for j in range(Mnew):
            for k in range(Mnew):
                soltrm[0,i*Mnew*Mnew+j*Mnew+k]=sol[0,(Mtrim+i)*M*M+(Mtrim+j)*M+Mtrim+k ]
    # the trimmed solution has been recorded.
    # Next we will check magnitudes of the truncated values and issue a warning if
    # Large values were truncated
    treshld=1.0e-2
    lrg_val_flg=False
    for i in range(M**3):
      # unwrap the index
      iu = i // (M*M)
      iv = (i-iu*M*M) // (M)
      iw =  i-iu*M*M-iv*M
      if (iu<Mtrim) or (iu > (M - 1 - Mtrim)) or (iv < Mtrim) or (iv > (M - 1 - Mtrim)) or (iw < Mtrim) or (iw > (M - 1 - Mtrim)):
          if numpy.abs(sol[0,i]) >= treshld:
               lrg_val_flg=True
    if lrg_val_flg:
      print('my_read_solution_trim: Attention truncating at least one large value. treshld=', treshld)
    return soltrm, nsze



##########################################################
# solution_untrim(soltrm,Mshrt,Mtrim)
#
# This subroutin pads the trimmed solution with zeros. Speficially,
# the one dimensional array is implicitly re-shaped as a 3D array with
# equal dimensions. Then 3D array is padded with Mtrim zeros in each index
# at the beginning and at the end. This operation undoes the trimming of the
# VDF solutions.
#
# soltrm -- the trimmed solution, shape (1,Mshft**3)
# Mshrt -- the dimension of the "trimmed" array
# Mtrim --- the number if index values to pad on both ends.
###########################################################
def solution_untrim(soltrm,Mshrt,Mtrim):
    # a quick check that the supplied values are consistens:
    import numpy as np
    if Mshrt ** 3 != soltrm.shape[0]:
        print('solution_untrim: the value Mshrt is not consistent with the trimmed solution size. Stop')
        exit()
    ####################
    M=Mshrt+2*Mtrim
    sol = np.zeros((1,M**3))
    for i in range(Mshrt**3):
        # unwrap the index
        iu = i // (Mshrt * Mshrt)
        iv = (i - iu * Mshrt * Mshrt) // (Mshrt)
        iw = i - iu * Mshrt * Mshrt - iv*Mshrt
        # make the long index
        ilong=(Mtrim+iu)*(M*M)+(Mtrim+iv)*M+(Mtrim+iw)
        sol[0,ilong] = soltrm[i]
    return sol, M**3
############################
# extract_time_name (path):
#
# This subroutine reads a path string and tries to extract the time stamp
# it looks for pattern: _timeX.XXXXXXXXXX_ and gets the X.XXXXXXXXX out
#
# Designed for names generated by the DGVlib code
# variable path has absolute path to the file to read including the file name
##############################
def extract_time_name (path):
    shift = path.find('_time')
    time_string = path[shift+5:shift+17]
    time_value=float(time_string)
    return time_value

############################
# my_read_solution_trim (path,M,Mtrim)
#
# This subroutine reads a solution and output of collision operator from the
# saved DGV0D3V *_SltnColl.dat file. It trims boundary cells and returns
# an array on a submehs inside of the domain. The submesh is formed by
# removing Mtrim cells on each end of the 1D velocity interval for ecah velocity variable
#
# we expect that the velocity mesh has the same number of cells in each dimension MxMxM
# Variables:
# path === pfull ath to solution file including the solution file name
# M -- size of the mesh used for the discrete solution
# Mtrim -- # cells to trim on each end of the 1D interval
#
#
##############################
def my_read_sol_coll_trim (path, M, Mtrim):
    #open binary file
    savefile=open(path, 'rb')

    #the first write is to record time to which this save corresponds.
    # the time variable is double precision and each write in fortran ads 32 bits before and after the written staff.
    # so 4 bytes+8 bytes+ 4bytes is 16
    bytes=savefile.read(16)
    # now that we got the first piece of information, let us unpack it using struct
    import struct
    data=struct.unpack('=ldl',bytes)
    scr1=data[0]
    t=data[1]
    scr2=data[2]
    #the second piece of information is the length of the solution array and the collision operator array,
    #not clear what the size of the array is, will try for long integer (4 bytes)
    # 4 bytes padding, 4 bytes the files size and 4 bytes
    bytes=savefile.read(12)
    data=struct.unpack('=lll',bytes)
    scr1 = data[0]    #Fortran writes the total length of data in bytes in a write statements in the beginning and in the end of the record
    fsize = data[1]
    scr2 = data[2]
    #next we need to read fsize double reals into a sequence (not sure what it will be, list or tuple
    bytes1 = savefile.read(4)
    two_arrays_len=struct.unpack('=l',bytes1)  # this should be the length of solution and collision operator arrays
    bytes = savefile.read(fsize*8)
    fmt='='+str(fsize)+"d"
    solscr=struct.unpack(fmt,bytes)
    # next we convert the sequence into a numpy array to return it.
    import numpy
    sol=numpy.array(solscr).reshape(1,fsize)
    #next we read the output of the collision operator
    bytes = savefile.read(fsize * 8)
    solscr = struct.unpack(fmt, bytes)
    coll = numpy.array(solscr).reshape(1,fsize)
    # debug
    #sh=sol.shape
    # end debug
    # close the file
    savefile.close()
    # The solution has been read from file into an array. Now we will need to trim it.
    # a quick check that the supplied values are consistens:
    if M**3 != fsize:
       print('my_read_solution_trim: the value M is not consistent with the solution size. Stop')
       exit()
    if 2*Mtrim > M:
       print('my_read_solution_trim: the provided value Mtrip is too large for the number of cells M. Stop')
       exit()
    ###################################
    # We are ready to trim the array/
    Mnew=M-2*Mtrim
    nsze=Mnew**3
    soltrm = numpy.zeros((1,nsze))
    colltrm = numpy.zeros((1,nsze))
    for i in range(Mnew):
        for j in range(Mnew):
            for k in range(Mnew):
                soltrm[0,i*Mnew*Mnew+j*Mnew+k]=sol[0,(Mtrim+i)*M*M+(Mtrim+j)*M+Mtrim+k ]
                colltrm[0,i*Mnew*Mnew+j*Mnew+k]=coll[0, (Mtrim + i) * M * M + (Mtrim + j) * M + Mtrim + k]
    # the trimmed solution and output of collision operator have been recorded.
    # Next we will check magnitudes of the truncated values and issue a warning if
    # Large values were truncated
    treshld=1.0e-3
    max_sol=numpy.amax(numpy.abs(sol[0,:]))
    max_coll = numpy.amax(numpy.abs(coll[0, :]))
    # figured the largest values in the arrays
    lrg_val_flg=False
    for i in range(fsize):
      # unwrap the index
      iu = i // (M*M)
      iv = (i-iu*M*M) // (M)
      iw =  i-iu*M*M-iv*M
      if (iu<Mtrim) or (iu > (M - 1 - Mtrim)) or (iv < Mtrim) or (iv > (M - 1 - Mtrim)) or (iw < Mtrim) or (iw > (M - 1 - Mtrim)):
          if numpy.abs(sol[0,i])>= treshld*max_sol or numpy.abs(coll[0,i]) >= treshld*max_coll :
               lrg_val_flg=True
    if lrg_val_flg:
      print('my_read_sol_coll_trim: Attention truncating at least one large value. treshld=', treshld)
    return soltrm, colltrm, nsze

############################################
# This subroutine will read the novdes and weigths arrays
# from a save file created by DGVlib
#
# ########################
def read_nodes(filename):
#open binary file
    savefile=open(filename, 'rb')

    #the first write is to record sizes of nodes arrays.
    # the size is 4 bytes integer in fortran ads 32 bits before and after the written staff.
    # so 4 bytes+4 bytes+ 4bytes is 12
    bytes=savefile.read(12)
    # now that we got the first piece of information, let us unpack it using struct
    import struct
    data=struct.unpack('=lll',bytes)
    scr1=data[0]
    nn=data[1]
    scr2=data[2]
    #the second piece of information is the length of the nodes arrays together,
    #not clear what the size of the array is, will try for long integer (4 bytes)
    #
    bytes=savefile.read(4)
    data=struct.unpack('=l',bytes)
    data_size = data[0]    #Fortran writes the total length of data in bytes in a write statements in the beginning and in the end of the record
    #next we read arrays nodes_u,_v,_w _gwts. each one should be nn long array of double precision
    bytes = savefile.read(nn*8)
    fmt='='+str(nn)+"d"
    arryscr=struct.unpack(fmt,bytes)
    # next we convert the sequence into a numpy array to return it.
    import numpy
    nodes_u = numpy.array(arryscr).reshape(1,nn)
    bytes = savefile.read(nn*8)
    arryscr = struct.unpack(fmt, bytes)
    nodes_v = numpy.array(arryscr).reshape(1,nn)
    bytes = savefile.read(nn*8)
    arryscr = struct.unpack(fmt, bytes)
    nodes_w = numpy.array(arryscr).reshape(1,nn)
    bytes = savefile.read(nn * 8)
    arryscr = struct.unpack(fmt, bytes)
    nodes_gwts = numpy.array(arryscr).reshape(1,nn)
    # close the file
    savefile.close()

    return nodes_u, nodes_v, nodes_w, nodes_gwts

#######################################
### save_moments(filename,moments_array)
###
### This subroutine write moments array on the hard drive in the csv format.
###
###
def save_moments(filename,moments_array):
    import numpy as np
    np.savetxt(filename, moments_array, delimiter=',')


############################################
# This subroutine will read the grids_arrays from the DGV solver format.
# from a save file created by DGVlib
#
# ########################
def read_grids(filename):
#open binary file
    savefile=open(filename, 'rb')

    #the first write is to record sizes of dgirs_cap_uvw arrays.
    # the size is 4 bytes integer in fortran ads 32 bits before and after the written staff.
    # so 4 bytes+4 bytes+ 4bytes is 12
    bytes=savefile.read(12)
    # now that we got the first piece of information, let us unpack it using struct
    import struct
    data=struct.unpack('=lll',bytes)
    scr1=data[0]
    nn=data[1]
    scr2=data[2]
    #the second piece of information is the three arrays grids_cap_u, grids_cap_v, grids_cap_w
    # they should have the same length, of typde double precision. So, one record should be 8 bytes
    #
    bytes=savefile.read(4)
    data=struct.unpack('=l',bytes)
    data_size = data[0]    #Fortran writes the total length of data in bytes in a write statements in the beginning and in the end of the record
    #next we read arrays grids_cap_u,_v,_w. each one should be nn long array of double precision
    bytes = savefile.read(nn*4) # read 1 array: grids_cap_u
    fmt='='+str(nn)+"l"
    arryscr=struct.unpack(fmt,bytes)
    # next we convert the sequence into a numpy array to return it.
    import numpy
    grids_cap_u = numpy.array(arryscr).reshape(1,nn)
    bytes = savefile.read(nn*4)
    arryscr = struct.unpack(fmt, bytes)
    grids_cap_v = numpy.array(arryscr).reshape(1,nn)
    bytes = savefile.read(nn*4)
    arryscr = struct.unpack(fmt, bytes)
    grids_cap_w = numpy.array(arryscr).reshape(1,nn)
    bytes=savefile.read(4) # read the concluding bite for the record.
    # end reading grids_cap_u,_v,_w arrrays.
    ################# Reading the number of records for grids_u
    bytes=savefile.read(12)
    # now that we got the first piece of information, let us unpack it using struct
    data=struct.unpack('=lll',bytes)
    scr1=data[0]
    nn=data[1]
    scr2=data[2]
    # now read the array grids_u
    bytes = savefile.read(4)
    data = struct.unpack('=l', bytes)
    data_size = data[0]  # Fortran writes the total length of data in bytes in a write statements in the beginning and in the end of the record
    # next we read arrays grids_u. should be nn long array of double precision
    bytes = savefile.read(nn * 8)  # read 1 array: grids_u
    fmt = '=' + str(nn) + "d"
    arryscr = struct.unpack(fmt, bytes)
    grids_u = numpy.array(arryscr).reshape(1, nn)
    bytes = savefile.read(4)
    # end reading grids_u
    ################# Reading the number of records for grids_v
    bytes = savefile.read(12)
    # now that we got the first piece of information, let us unpack it using struct
    data = struct.unpack('=lll', bytes)
    scr1 = data[0]
    nn = data[1]
    scr2 = data[2]
    # now read the array grids_v
    bytes = savefile.read(4)
    data = struct.unpack('=l', bytes)
    data_size = data[0]  # Fortran writes the total length of data in bytes in a write statements in the beginning and in the end of the record
    # next we read arrays grids_u. should be nn long array of double precision
    bytes = savefile.read(nn * 8)  # read 1 array: grids_v
    fmt = '=' + str(nn) + "d"
    arryscr = struct.unpack(fmt, bytes)
    grids_v = numpy.array(arryscr).reshape(1, nn)
    bytes = savefile.read(4)
    # end reading grids_v
    ################# Reading the number of records for grids_w
    bytes = savefile.read(12)
    # now that we got the first piece of information, let us unpack it using struct
    data = struct.unpack('=lll', bytes)
    scr1 = data[0]
    nn = data[1]
    scr2 = data[2]
    # now read the array grids_w
    bytes = savefile.read(4)
    data = struct.unpack('=l', bytes)
    data_size = data[0]  # Fortran writes the total length of data in bytes in a write statements in the beginning and in the end of the record
    # next we read arrays grids_u. should be nn long array of double precision
    bytes = savefile.read(nn * 8)  # read 1 array: grids_w
    fmt = '=' + str(nn) + "d"
    arryscr = struct.unpack(fmt, bytes)
    grids_w = numpy.array(arryscr).reshape(1, nn)
    bytes = savefile.read(4)
    # end reading grids_u
    savefile.close()

    return grids_cap_u, grids_cap_v, grids_cap_w, grids_u, grids_v, grids_w
