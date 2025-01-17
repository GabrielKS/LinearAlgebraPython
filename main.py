# LinearAlgebraPython, by Gabriel Konar-Steenberg
# An attempt to write out in code some linear algebra algorithms and properties, mostly for my own educational benefit
#
# Technical correctness is a high priority; optimization is very much not. Elegance is nice. Things should be Pythonic where that is nice.
# I use the built-in Python libraries, but no external libraries -- in particular, no NumPy.
#
# From time to time I cite my first and second linear algebra textbooks with an abbreviation and a section or page number; these textbooks are, respectively:
#   B:  Bretscher, Otto (2013). "Linear Algebra with Applications (Fifth Edition)."
#   GH: Garcia, Stephan Ramon and Horn, Roger A. (2017). "A Second Course in Linear Algebra."
#
# Conventions:
#   - I always write "col" instead of "column"
#   - Everything is indexed from 0, so the top-left entry of a matrix is mat[0,0]
#   - The 0x0 matrix exists and its representation in array form is [] (not [[]])
#   - An n-dimensional vector is a nx1 matrix (meaning its representation in array form is [[v1],[v2],[v3],...])
#
# Major TODO items (minor items are scattered throughout the code):
#   - TODO: Handle fractions nicely using Python's native fraction support
#   - TODO: Implement automatic fraction handling to avoid annoying floating point errors (2.999999...8)
#   - TODO: Split the Matrix class off into a matrix.py and just put tests here
#   - TODO: Come up with a better name for the project
#   - TODO: I get the sense that the various methods of iteration and enumeration could use some cleanup
#   - TODO: Maybe implement a more standardized testing protocol

import time
import numbers
import copy
import decimal
import math
from random import random
import operator

def main():
    # Real vectors (are matrices)
    v = Matrix.vector_from_list([1, 2, 3])
    # print(v)

    # Complex vectors
    u = Matrix.vector_from_list([4+1j, 5-1j, 6])
    # print(u)

    # Real matrices
    m = Matrix([[1, 2],[3, 4]])
    # print(m)

    # Complex matrices
    n = Matrix([[1+2j, 3+4j], [5+6j, 7+8j]])
    # print(n)

    # Nice subscripting, including submatrices
    # print(m[0,0])
    # print(m[0,:])
    # print(m[:,0])
    m1 = m.copy()
    m1[0,0] = -1
    # print(m1)
    m1[0, :] = Matrix([[-2, -3]])
    # print(m1)
    m1[:, 0] = Matrix([[-4],[-5]])
    # print(m1)

    # Transposition
    a = Matrix([[1,2,3],[4,5,6]])
    # print(a)
    # print(a.transpose())

    # # Dot product
    # print(u)
    # print(v.dot(u))

    # # Determinant
    c = Matrix([[42, 1, 0], [-5, 1, 0], [0, 0, 1]])
    # print(c.det())
    # print(Matrix([[3,1,4],[1,5,9],[2,6,5]]).det())

    #Elementary operations
    # print(-m)
    # print(m+n)
    # print(m-n)
    # print(5*m)
    # print(m*5)
    # print(m*n)
    # print(m/4)

    #Enumeration
    # for (i,j),x in a.enumerate():
        # print("["+str(i)+","+str(j)+"]: "+str(x))

    #Identity matrix
    # print(Matrix.ident(5))

    #Equality
    # print(Matrix([[5]]) == Matrix([[5]]))
    # print(Matrix([[5, 6]]) == Matrix([[5]]))
    # print(Matrix([[6]]) == Matrix([[5]]))

    #Block matrices
    block = Matrix.from_block([                      # [1 1|2]
        [Matrix([[1,1],[1,1]]), Matrix([[2],[2]])],  # [1_1|2]
        [Matrix([[3,3]]), 4]                         # [3 3|4]
    ])
    # print(block)

    #Sorting of rows or cols
    # print(Matrix([[1,2,3],[4,5,6],[7,8,9]]).sort(lambda r: random(), dim="rows"))

    #rref and row operations
    # print(a.rref())
    r = Matrix.construct_by_indices(lambda i,j: random(), 5, 6)
    # print(r)
    # print(r.rref())
    g = Matrix([
        [1, 2, 0, 0, 3],
        [0, 0, 0, 1, 1],
        [0, 0, 3, 0, 9],
        [0, 0, 0, 0, 0]
    ])
    g[0] += 2*g[1]
    g[1] += 2*g[0]
    g[2] += 2*g[0]
    g[3] += 2*g[2] -2*g[1] +4*g[0]
    # print(g)
    # print(g.rref())

    #Orthonormality
    # print(Matrix([[1,0],[0,1]]).is_orthonormal())  # orthonormal
    # print(Matrix([[1,0],[1,0]]).is_orthonormal())  # not orthogonal
    # print(Matrix([[2,0],[0,1]]).is_orthonormal())  # not normal
    # print(Matrix([[1/math.sqrt(2),1/math.sqrt(2)],[1/math.sqrt(2),-1/math.sqrt(2)]]).is_orthonormal(tolerance = 1e-15))  # orthonormal, but only with a tolerance

    #Gram-Schmidt
    gs = Matrix([[3,1,4],[2,7,1],[1,6,1]]).gram_schmidt()
    # print(gs)
    # print(gs.is_orthonormal(tolerance=1e-10))

    #Householder
    v = Matrix.vector_from_list([4,0,-3])
    print(v.householder())
    # ua = v.unitary_annihilator()
    # print(v)
    # print(ua)
    # print(ua*v)

    #QR Factorization
    s = Matrix([[3,1],[4,2]])
    q,r = s.qr_factorization(tolerance=0.01)
    # print(s)
    # print(q)
    # print(r)
    # print(q*r)




Scalar = numbers.Complex  # GH 0.2
Real = numbers.Real

def kronecker_delta(i, j):  # GH pg3
    return 1 if i == j else 0

def is_2d_non_jagged_of_type(arr, type):
    if arr == [[]]: return False  # We adopt the convention that a 0x0 2d matrix is [], not [[]]
    if arr == []: return True
    for row in arr:
        if len(row) != len(arr[0]): return False  # Ensures that the array is not jagged
        for value in row:
            if not isinstance(value, type): return False  # Ensures that all elements are numbers
    return True

class Matrix():  # GH 0.3
    # STATE
    # A Matrix's state is defined entirely by the following:
    #   - values: a list of 0 or more lists of entries
    #   - col_major: a bool representing whether the top-level entries of values are rows (False) or columns (True)

    # CREATION AND MUTATION
    def __init__(self, values, col_major=False):  # The main way to create a matrix is by providing a two-dimensional array (i.e., a list of lists) with the desired values. Other ways include construction as a block matrix, construction by index, and using the built-in methods for zero matrices and identity matrices.
        self.reset(values, col_major)
    
    def reset(self, values, col_major=False):  # Re-initialize
        if isinstance(values, Scalar): self.reset([[values]])  # Tolerate scalars by converting them to 1x1 matrices
        else:
            assert is_2d_non_jagged_of_type(values, Scalar)
            self.values = copy.deepcopy(values)
            self.col_major = col_major

    def assign(self, other):  # Mutate self into other (i.e., make self equal to other without messing with references)
        self.reset(other.values, other.col_major)
        assert self == other
    
    @classmethod
    def from_block(cls, submatrices):  # Assumes that the blocks line up, i.e., all the blocks on a given input row (resp. col) have the same number of rows (resp. cols)
        for submatrices_row in range(len(submatrices)):
            for submatrices_col in range(len(submatrices[submatrices_row])):
                if isinstance(submatrices[submatrices_row][submatrices_col], Scalar):
                    submatrices[submatrices_row][submatrices_col] = Matrix(submatrices[submatrices_row][submatrices_col])  # Tolerate scalars by converting them to 1x1 matrices

        assert is_2d_non_jagged_of_type(submatrices, cls)
        submatrices_rows = len(submatrices)
        submatrices_cols = len(submatrices[0]) if submatrices_rows > 0 else 0
        result = cls.zero(sum([submatrices[i][0].rows for i in range(submatrices_rows)]), sum([submatrices[0][j].cols for j in range(submatrices_cols)]))
        for submatrices_row in range(len(submatrices)):
            for submatrices_col in range(len(submatrices[submatrices_row])):
                submatrix = submatrices[submatrices_row][submatrices_col]
                result_row_offset = sum([submatrices[i][0].rows for i in range(submatrices_row)])
                result_col_offset = sum([submatrices[0][j].cols for j in range(submatrices_col)])
                for (i,j),x in submatrix.enumerate():
                    result[result_row_offset+i, result_col_offset+j] = submatrix[i,j]
        return result

    @classmethod
    def vector_from_list(cls, values):
        return cls([values]).transpose()
    
    def vector_to_list(self):
        assert self.is_vector()
        return self.values[0] if col_major else self.transpose().values[0]
    
    def is_vector(self):
        return self.cols == 1
    
    def unpack_index_key(self, key): # Basically i,j=key with lots of error handling and special cases
        try:
            i,j = key
            col_present = True
        except TypeError:
            i,j = (key, slice(None, None, None))  # Shortcut for rows: you can do mat[3] instead of mat[3,:]
            col_present = False

        row_malformed = not(isinstance(i, int) or isinstance(i, slice))
        col_malformed = not(isinstance(j, int) or isinstance(j, slice))
        if row_malformed or col_malformed:
            if row_malformed and not col_present: msg = "Malformed key: key is "+str(type(key))+"; should be int|slice or (int|slice, int|slice) tuple"

            elif row_malformed: msg = "Malformed row index: row index is "+str(type(i))+"; should be int|slice."
            else: msg = "Row OK."

            if col_malformed: msg += " Malformed col index: col index is "+str(type(i))+"; should be int|slice."
            else: msg = "Col OK."

            raise TypeError(msg)
        
        return i,j

    def __getitem__(self, key):
        i,j = self.unpack_index_key(key)
        if self.col_major: i,j = j,i
        if isinstance(i, slice) or isinstance(j, slice):
            sliced_rows = self.values[i]
            if len(sliced_rows) == 0: return []
            if type(sliced_rows[0]) is not list: sliced_rows = [sliced_rows]
            result = Matrix([(row[j] if type(row[j]) is list else [row[j]]) for row in sliced_rows])
            # if result.rows == 1 and result.cols == 1: return result[0,0]  # If the result is a single value, return the value instead of a 1x1 matrix
            return result
        else:
            return self.values[i][j]
    
    def __setitem__(self, key, value):
        i,j = self.unpack_index_key(key)
        if self.col_major: i,j = j,i
        if isinstance(value, Matrix): value = value.values
        if isinstance(i, slice) or isinstance(j, slice):
            sliced_rows = self.values[i]
            if len(sliced_rows) == 0: return []
            if type(sliced_rows[0]) is not list: sliced_rows = [sliced_rows]
            for x, row in enumerate(sliced_rows):
                row[j] = value[x]
        else:
            self.values[i][j] = value
    
    def copy(self):
        return copy.deepcopy(self)
    
    #PROPERTIES
    @property
    def rows(self):
        return (len(self.values[0]) if self.cols > 0 else 0) if self.col_major else len(self.values)
    
    @property
    def cols(self):
        return len(self.values) if self.col_major else (len(self.values[0]) if self.rows > 0 else 0)
    
    @property
    def length(self):
        return self.rows
    
    def __len__(self):  # WARNING: len(mat) will return the number of rows, not the number of entries
        return self.len
    
    @property
    def len(self):
        return self.rows

    def __repr__(self):
        return "Matrix("+str(self.values)+(", col_major=True" if self.col_major else "")+")"
    
    def __str__(self):
        return "["+" \n ".join(["["+", ".join([str(v) for v in row])+"]" for row in self.values])+"]"+("T" if self.col_major else "")
    
    #ELEMENT MANIPULATION
    # TODO: come up with a more generalized way of doing inplace
    def __iter__(self):
        return self.MatrixIterator(self)
    
    class MatrixIterator():
        def __init__(self, matrix):
            self.matrix = matrix
            self.row = 0
            self.col = 0
        
        def __next__(self):
            try: result = self.matrix[self.row, self.col]
            except IndexError: raise StopIteration
            self.col += 1
            if self.col >= self.matrix.cols:
                self.col = 0
                self.row += 1
            return result
    
    def enumerate(self):  # Iterate through the rows in the order (1, 1), (1, 2), ..., (1, cols-1), (2, 1), (2, 2), ..., (rows-1, cols-1) and return ((row, col), value) at each point
        for i in range(self.rows):
            for j in range(self.cols):
                yield ((i, j), self[i,j])
    
    def enumrows(self):
        for i in range(self.rows):
            yield (i, self[i,:])
        
    def enumcols(self):
        for j in range(self.cols):
            yield (j, self[:,j])
        
    def sortrows(self, key, inplace=False):
        return self.sort(key, "rows", inplace)
    
    def sortcols(self, key, inplace=False):
        return self.sort(key, "cols", inplace)
    
    def sort(self, key, dim, inplace=False):  # Sort the rows or cols of a matrix by key, where key is a function that can be applied to a row or col matrix to get something sortable
        if inplace:
            self.assign(self.sort(key, dim, inplace=False))
        else:
            assert dim == "rows" or dim == "cols"
            l = [r for _,r in self.enumrows()] if dim == "rows" else [c for _,c in self.enumcols()]
            l = sorted(l, key=key)
            return Matrix.from_block([[r] for r in l]) if dim == "rows" else Matrix.from_block([l])

    #TODO: maybe implement a swap method for row/col swapping

    def apply(self, f, inplace=False):
        if not inplace: 
            result = self.copy()
            result.apply(f, inplace=True)
            return result
        else:
            for (i,j),x in self.enumerate():
                    self[i,j] = f(x)
    
    def apply_binary(self, other, f, inplace=False):
        if not inplace:
            result = self.copy()
            result.apply_binary(other, f, inplace=True)
            return result
        else:
            for (i,j),x in self.enumerate():
                self[i,j] = f(x, other[i,j])
        
    
    @classmethod
    def construct_by_indices(cls, f, rows, cols):  # Given a function f that takes a set of row,col indices and returns an entry, this method creates a matrix of the desired size
        result = cls.zero(rows, cols)
        for (i,j),_ in result.enumerate():
            result[i,j] = f(i,j)
        return result
    
    #UNARY OPERATIONS
    def __pos__(self): return self.apply(operator.pos)
    def __neg__(self): return self.apply(operator.neg)
    # def __abs__(self): return self.apply(operator.abs)

    def transpose(self):  # TODO: inplace
        result = self.zero(self.cols, self.rows)
        for (i,j),x in self.enumerate():
                result[j,i] = x
        return result
    
    def conjugate(self, inplace=False):
        return self.apply(lambda x: x.conjugate(), inplace)
    
    def adjoint(self):  # TODO: inplace
        return self.transpose().conjugate()
    
    def det(self):  # Computes the determinant by cofactor expansion about the first (row if self.col_major else col). TODO rewrite so it is easier to follow and doesn't refer to self.values
        assert self.rows == self.cols
        if self.rows == 0:
            return 1
        result = 0
        for i in range(self.rows):
            sub = Matrix([[e for c, e in enumerate(row) if c != 0] for r, row in enumerate(self.values) if r != i])
            d = sub.det()
            x = ((-1)**i)*self[i,0]*d
            result += x
        return result
    
    def leading(self):  # Returns (row, col), value for the leading value. This is the leading (i.e., first nonzero) value in the first nonzero row.
        for (i,j),x in self.enumerate():  # enumerate() is guaranteed to iterate the way we want it to
            if x != 0:
                return ((i,j),x)
        return ((None,None),None)

    def mag(self):  # Magnitude
        return math.sqrt(self.dot(self))

    #BINARY OPERATIONS
    def __eq__(self, other): return self.rows == other.rows and self.cols == other.cols and all([x == other[i,j] for (i,j),x in self.enumerate()])

    def __add__(self, other): return self.apply_binary(other, operator.add)
    def __sub__(self, other): return self.apply_binary(other, operator.sub)

    def dot(self, other):
        assert self.is_vector()
        assert other.is_vector()
        return sum(self.apply_binary(other, lambda x,y: x*y))

    def __mul__(self, other):
        if isinstance(other, Matrix) and other.rows == 1 and other.cols == 1:
            return self*other[0,0]
        if isinstance(other, Scalar):
            return self.apply(lambda x: other*x)
        else:
            assert self.cols == other.rows
            result = self.zero(self.rows, other.cols)
            for (i,j),_ in result.enumerate():
                result[i,j] = self[i,:].transpose().dot(other[:,j])
            return result

    def __rmul__(self, other):
        assert isinstance(other, Scalar) or (isinstance(other, Matrix) and other.rows == 1 and other.cols == 1)
        return self*other
    
    def __truediv__(self, other):
        assert isinstance(other, Scalar)
        return self.apply(lambda x: x/other)
    
    def direct_sum(self, other):
        return self.from_block([[self, self.zero(self.rows, other.cols)],[self.zero(other.rows, self.cols), other]])

    #SPECIAL MATRICES
    @classmethod
    def zero(cls, rows, cols):
        if rows == 0 and cols > 0: return cls([[] for i in range(cols)], col_major=True)
        else: return cls([[0] * cols for i in range(rows)])
    
    @classmethod
    def ident(cls, n):  # GH pg3
        return cls.construct_by_indices(kronecker_delta, n, n)
    
    @classmethod
    def standard_basis(cls, rows, i):
        return cls.construct_by_indices(lambda row,col: 1 if row == i else 0, rows, 1)
    
    #TESTS FOR PROPERTIES
    def is_zero(self, tolerance=0):
        # return all([x == 0 for x in self])
        return sum(self.apply(lambda x: x*x)) <= tolerance  # A more tolerant method: a matrix is zero if the sum of the squares of the entries is <= tolerance. This is mathematically correct if tolerance=0.

    def is_identity(self):
        return all([x == kronecker_delta(i,j) for (i,j),x in self.enumerate()])

    def is_inverse(self, other):
        return (self*other).is_identity()

    def is_orthonormal(self, tolerance=0):  # GH 5.1.2. WARNING: in almost all practical cases, floating point imprecision will require you to set a nonzero tolerance to get useful results. TODO: determine whether there can be a good and fairly universal nonzero tolerance value.
        for i1,row1 in self.enumcols():
            for i2,row2 in self.enumcols():
                if abs(row1.dot(row2)-kronecker_delta(i1, i2)) > tolerance: return False
        return True
    
    def is_upper_triangular(self, tolerance=0):
        for (i,j),x in self.enumerate():
            if j<i and abs(x) > tolerance: return False
        return True
    
    def is_lower_triangular(self, tolerance=0):
        for (i,j),x in self.enumerate():
            if j>i and abs(x) > tolerance: return False
        return True
    
    def is_triangular(self, tolerance=0):
        return self.is_upper_triangular(tolerance) and self.is_lower_triangular(tolerance)
    
    #ALGORITHMS
    def rref(self, inplace=False):  # B pg15
        if not inplace:
            result = self.copy()
            result.rref(inplace=True)
            return result
        else:
            for i,row in self.enumrows():
                # Find the leading value in row i
                (_,leading_index),leading_value = row.leading()

                # Normalize the leading value, if possible
                if leading_index is None: break
                else: self[i,:] /= leading_value

                # Eliminate the rest of the values in col leading_index
                for i_e, row_e in self.enumrows():
                    if i_e != i:
                        self[i_e,:] -= self[i,:]*row_e[0,leading_index]
        
        # Sort the rows so that leading values go from left to right as row number increases
        # We do this with a nested lambda expression that returns the index of the leading value if it exists, otherwise the number of cols in the row. Nesting allows us to only call leading() once per row.
        self.sortrows(lambda row: (lambda row, leading: leading if leading is not None else row.cols)(row, row.leading()[0][1]), inplace=True)

    def gram_schmidt(self, inplace=False):  # GH pg107. Applies the Gram-Schmidt process on the cols of self to create an orthonormal basis. Assumes cols are linearly independent.
        if not inplace:
            result = self.copy()
            result.gram_schmidt(inplace=True)
            return result
        else:
            for j_active,col_active in self.enumcols():
                for j_previous,col_previous in self.enumcols():
                    if j_previous >= j_active: break
                    self[:,j_active] -= col_active.dot(col_previous)*col_previous
                self[:,j_active] /= self[:,j_active].mag()
                # TODO: add an is_orthonormal assertion once I've figured out if there can be a good value for tolerance

    def householder(self):  # GH 6.4.3
        assert self.is_vector()
        u = self/self.mag()
        pu = u*u.adjoint()
        return self.ident(len(self))-2*pu
    
    def unitary_annihilator(self, tolerance=0):  # Full name: "unitary annihilator of the lower entries." GH 6.4.10. Returns a matrix such that x.unitary_annihilator()*x == x.mag()*Matrix.standard_basis(len(x), 1) (to within a tolerance)
        assert self.is_vector()
        sigma = 1 if abs(self[0,0]) <= tolerance else -(self[0,0]/abs(self[0,0]))
        return sigma*self.householder()
    
    def qr_factorization(self, tolerance=0, flavor="narrow"):  # GH 6.5.2
        m = self.rows
        n = self.cols
        assert m >= n
        a = self
        u = self.ident(m)

        ap = a
        for j in range(n):
            a1 = ap[:,0]
            up = ap.ident(m) if a1.is_zero(tolerance) else a1.householder()
            uj = self.ident(j).direct_sum(up)
            ap = (uj*ap)[1:, 1:]
            u = uj*u

        r = u*a
        v = u.adjoint()

        if flavor == "wide":
            return v, Matrix.from_block([[r],[Matrix.zero(v.cols-r.rows, r.cols)]])
        elif flavor == "narrow":
            q = v[:,:n]
            return q, r


if __name__ == "__main__":
    main()