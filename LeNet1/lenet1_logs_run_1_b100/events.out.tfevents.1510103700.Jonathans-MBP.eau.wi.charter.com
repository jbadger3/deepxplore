       �K"	   ����Abrain.Event:2����     JB�	u"����A"��
`
lenet1/inputsPlaceholder*(
_output_shapes
:����������*
shape: *
dtype0
m
lenet1/Reshape/shapeConst*
_output_shapes
:*%
valueB"����         *
dtype0
�
lenet1/ReshapeReshapelenet1/inputslenet1/Reshape/shape*
T0*/
_output_shapes
:���������*
Tshape0
_
lenet1/labelsPlaceholder*'
_output_shapes
:���������
*
shape: *
dtype0
�
1lenet1/h1/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@lenet1/h1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
/lenet1/h1/kernel/Initializer/random_uniform/minConst*#
_class
loc:@lenet1/h1/kernel*
valueB
 *�X`�*
dtype0*
_output_shapes
: 
�
/lenet1/h1/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@lenet1/h1/kernel*
valueB
 *�X`>*
dtype0*
_output_shapes
: 
�
9lenet1/h1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1lenet1/h1/kernel/Initializer/random_uniform/shape*#
_class
loc:@lenet1/h1/kernel*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:
�
/lenet1/h1/kernel/Initializer/random_uniform/subSub/lenet1/h1/kernel/Initializer/random_uniform/max/lenet1/h1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
/lenet1/h1/kernel/Initializer/random_uniform/mulMul9lenet1/h1/kernel/Initializer/random_uniform/RandomUniform/lenet1/h1/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
+lenet1/h1/kernel/Initializer/random_uniformAdd/lenet1/h1/kernel/Initializer/random_uniform/mul/lenet1/h1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
lenet1/h1/kernel
VariableV2*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
lenet1/h1/kernel/AssignAssignlenet1/h1/kernel+lenet1/h1/kernel/Initializer/random_uniform*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
lenet1/h1/kernel/readIdentitylenet1/h1/kernel*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
 lenet1/h1/bias/Initializer/ConstConst*!
_class
loc:@lenet1/h1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/h1/bias
VariableV2*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
lenet1/h1/bias/AssignAssignlenet1/h1/bias lenet1/h1/bias/Initializer/Const*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
w
lenet1/h1/bias/readIdentitylenet1/h1/bias*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:
t
lenet1/h1/convolution/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0
t
#lenet1/h1/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
lenet1/h1/convolutionConv2Dlenet1/Reshapelenet1/h1/kernel/read*/
_output_shapes
:���������*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides

�
lenet1/h1/BiasAddBiasAddlenet1/h1/convolutionlenet1/h1/bias/read*
T0*/
_output_shapes
:���������*
data_formatNHWC
c
lenet1/h1/ReluRelulenet1/h1/BiasAdd*
T0*/
_output_shapes
:���������
�
lenet1/h2/AvgPoolAvgPoollenet1/h1/Relu*/
_output_shapes
:���������*
paddingVALID*
ksize
*
T0*
data_formatNHWC*
strides

�
1lenet1/h3/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@lenet1/h3/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
/lenet1/h3/kernel/Initializer/random_uniform/minConst*#
_class
loc:@lenet1/h3/kernel*
valueB
 *����*
dtype0*
_output_shapes
: 
�
/lenet1/h3/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@lenet1/h3/kernel*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
9lenet1/h3/kernel/Initializer/random_uniform/RandomUniformRandomUniform1lenet1/h3/kernel/Initializer/random_uniform/shape*#
_class
loc:@lenet1/h3/kernel*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:
�
/lenet1/h3/kernel/Initializer/random_uniform/subSub/lenet1/h3/kernel/Initializer/random_uniform/max/lenet1/h3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@lenet1/h3/kernel*
_output_shapes
: 
�
/lenet1/h3/kernel/Initializer/random_uniform/mulMul9lenet1/h3/kernel/Initializer/random_uniform/RandomUniform/lenet1/h3/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
+lenet1/h3/kernel/Initializer/random_uniformAdd/lenet1/h3/kernel/Initializer/random_uniform/mul/lenet1/h3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
lenet1/h3/kernel
VariableV2*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
lenet1/h3/kernel/AssignAssignlenet1/h3/kernel+lenet1/h3/kernel/Initializer/random_uniform*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
lenet1/h3/kernel/readIdentitylenet1/h3/kernel*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
 lenet1/h3/bias/Initializer/ConstConst*!
_class
loc:@lenet1/h3/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/h3/bias
VariableV2*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
lenet1/h3/bias/AssignAssignlenet1/h3/bias lenet1/h3/bias/Initializer/Const*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
w
lenet1/h3/bias/readIdentitylenet1/h3/bias*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:
t
lenet1/h3/convolution/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0
t
#lenet1/h3/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
�
lenet1/h3/convolutionConv2Dlenet1/h2/AvgPoollenet1/h3/kernel/read*/
_output_shapes
:���������*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides

�
lenet1/h3/BiasAddBiasAddlenet1/h3/convolutionlenet1/h3/bias/read*
T0*/
_output_shapes
:���������*
data_formatNHWC
c
lenet1/h3/ReluRelulenet1/h3/BiasAdd*
T0*/
_output_shapes
:���������
�
lenet1/h4/AvgPoolAvgPoollenet1/h3/Relu*/
_output_shapes
:���������*
paddingVALID*
ksize
*
T0*
data_formatNHWC*
strides

g
lenet1/Reshape_1/shapeConst*
_output_shapes
:*
valueB"�����   *
dtype0
�
lenet1/Reshape_1Reshapelenet1/h4/AvgPoollenet1/Reshape_1/shape*
T0*(
_output_shapes
:����������*
Tshape0
�
6lenet1/outputs/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@lenet1/outputs/kernel*
valueB"�   
   *
dtype0*
_output_shapes
:
�
4lenet1/outputs/kernel/Initializer/random_uniform/minConst*(
_class
loc:@lenet1/outputs/kernel*
valueB
 *W{0�*
dtype0*
_output_shapes
: 
�
4lenet1/outputs/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@lenet1/outputs/kernel*
valueB
 *W{0>*
dtype0*
_output_shapes
: 
�
>lenet1/outputs/kernel/Initializer/random_uniform/RandomUniformRandomUniform6lenet1/outputs/kernel/Initializer/random_uniform/shape*(
_class
loc:@lenet1/outputs/kernel*

seed *
seed2 *
dtype0*
T0*
_output_shapes
:	�

�
4lenet1/outputs/kernel/Initializer/random_uniform/subSub4lenet1/outputs/kernel/Initializer/random_uniform/max4lenet1/outputs/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
: 
�
4lenet1/outputs/kernel/Initializer/random_uniform/mulMul>lenet1/outputs/kernel/Initializer/random_uniform/RandomUniform4lenet1/outputs/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
0lenet1/outputs/kernel/Initializer/random_uniformAdd4lenet1/outputs/kernel/Initializer/random_uniform/mul4lenet1/outputs/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
lenet1/outputs/kernel
VariableV2*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
shape:	�
*
	container *
dtype0*
shared_name 
�
lenet1/outputs/kernel/AssignAssignlenet1/outputs/kernel0lenet1/outputs/kernel/Initializer/random_uniform*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
lenet1/outputs/kernel/readIdentitylenet1/outputs/kernel*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
%lenet1/outputs/bias/Initializer/ConstConst*&
_class
loc:@lenet1/outputs/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
lenet1/outputs/bias
VariableV2*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
shape:
*
	container *
dtype0*
shared_name 
�
lenet1/outputs/bias/AssignAssignlenet1/outputs/bias%lenet1/outputs/bias/Initializer/Const*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
lenet1/outputs/bias/readIdentitylenet1/outputs/bias*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:

�
lenet1/outputs/MatMulMatMullenet1/Reshape_1lenet1/outputs/kernel/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
lenet1/outputs/BiasAddBiasAddlenet1/outputs/MatMullenet1/outputs/bias/read*
T0*'
_output_shapes
:���������
*
data_formatNHWC
k
lenet1/outputs/SigmoidSigmoidlenet1/outputs/BiasAdd*
T0*'
_output_shapes
:���������

�
$lenet1/global_step/Initializer/ConstConst*%
_class
loc:@lenet1/global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
lenet1/global_step
VariableV2*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
shape: *
	container *
dtype0	*
shared_name 
�
lenet1/global_step/AssignAssignlenet1/global_step$lenet1/global_step/Initializer/Const*
T0	*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
validate_shape(*
use_locking(

lenet1/global_step/readIdentitylenet1/global_step*
T0	*%
_class
loc:@lenet1/global_step*
_output_shapes
: 
M
lenet1/RankConst*
_output_shapes
: *
value	B :*
dtype0
b
lenet1/ShapeShapelenet1/outputs/Sigmoid*
T0*
_output_shapes
:*
out_type0
O
lenet1/Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
d
lenet1/Shape_1Shapelenet1/outputs/Sigmoid*
T0*
_output_shapes
:*
out_type0
N
lenet1/Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
O

lenet1/SubSublenet1/Rank_1lenet1/Sub/y*
T0*
_output_shapes
: 
`
lenet1/Slice/beginPack
lenet1/Sub*
T0*

axis *
_output_shapes
:*
N
[
lenet1/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
~
lenet1/SliceSlicelenet1/Shape_1lenet1/Slice/beginlenet1/Slice/size*
T0*
_output_shapes
:*
Index0
i
lenet1/concat/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
T
lenet1/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
lenet1/concatConcatV2lenet1/concat/values_0lenet1/Slicelenet1/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
�
lenet1/Reshape_2Reshapelenet1/outputs/Sigmoidlenet1/concat*
T0*0
_output_shapes
:������������������*
Tshape0
O
lenet1/Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
[
lenet1/Shape_2Shapelenet1/labels*
T0*
_output_shapes
:*
out_type0
P
lenet1/Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
S
lenet1/Sub_1Sublenet1/Rank_2lenet1/Sub_1/y*
T0*
_output_shapes
: 
d
lenet1/Slice_1/beginPacklenet1/Sub_1*
T0*

axis *
_output_shapes
:*
N
]
lenet1/Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
�
lenet1/Slice_1Slicelenet1/Shape_2lenet1/Slice_1/beginlenet1/Slice_1/size*
T0*
_output_shapes
:*
Index0
k
lenet1/concat_1/values_0Const*
_output_shapes
:*
valueB:
���������*
dtype0
V
lenet1/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
lenet1/concat_1ConcatV2lenet1/concat_1/values_0lenet1/Slice_1lenet1/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N
�
lenet1/Reshape_3Reshapelenet1/labelslenet1/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
$lenet1/SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitslenet1/Reshape_2lenet1/Reshape_3*
T0*?
_output_shapes-
+:���������:������������������
P
lenet1/Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
Q
lenet1/Sub_2Sublenet1/Ranklenet1/Sub_2/y*
T0*
_output_shapes
: 
^
lenet1/Slice_2/beginConst*
_output_shapes
:*
valueB: *
dtype0
c
lenet1/Slice_2/sizePacklenet1/Sub_2*
T0*

axis *
_output_shapes
:*
N
�
lenet1/Slice_2Slicelenet1/Shapelenet1/Slice_2/beginlenet1/Slice_2/size*
T0*#
_output_shapes
:���������*
Index0
�
lenet1/Reshape_4Reshape$lenet1/SoftmaxCrossEntropyWithLogitslenet1/Slice_2*
T0*#
_output_shapes
:���������*
Tshape0
V
lenet1/ConstConst*
_output_shapes
:*
valueB: *
dtype0
q
lenet1/MeanMeanlenet1/Reshape_4lenet1/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
\
lenet1/loss/tagsConst*
_output_shapes
: *
valueB Blenet1/loss*
dtype0
\
lenet1/lossScalarSummarylenet1/loss/tagslenet1/Mean*
T0*
_output_shapes
: 
Y
lenet1/gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
[
lenet1/gradients/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
n
lenet1/gradients/FillFilllenet1/gradients/Shapelenet1/gradients/Const*
T0*
_output_shapes
: 
y
/lenet1/gradients/lenet1/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
)lenet1/gradients/lenet1/Mean_grad/ReshapeReshapelenet1/gradients/Fill/lenet1/gradients/lenet1/Mean_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
w
'lenet1/gradients/lenet1/Mean_grad/ShapeShapelenet1/Reshape_4*
T0*
_output_shapes
:*
out_type0
�
&lenet1/gradients/lenet1/Mean_grad/TileTile)lenet1/gradients/lenet1/Mean_grad/Reshape'lenet1/gradients/lenet1/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
y
)lenet1/gradients/lenet1/Mean_grad/Shape_1Shapelenet1/Reshape_4*
T0*
_output_shapes
:*
out_type0
l
)lenet1/gradients/lenet1/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
q
'lenet1/gradients/lenet1/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
&lenet1/gradients/lenet1/Mean_grad/ProdProd)lenet1/gradients/lenet1/Mean_grad/Shape_1'lenet1/gradients/lenet1/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
s
)lenet1/gradients/lenet1/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
(lenet1/gradients/lenet1/Mean_grad/Prod_1Prod)lenet1/gradients/lenet1/Mean_grad/Shape_2)lenet1/gradients/lenet1/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
m
+lenet1/gradients/lenet1/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
�
)lenet1/gradients/lenet1/Mean_grad/MaximumMaximum(lenet1/gradients/lenet1/Mean_grad/Prod_1+lenet1/gradients/lenet1/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
*lenet1/gradients/lenet1/Mean_grad/floordivFloorDiv&lenet1/gradients/lenet1/Mean_grad/Prod)lenet1/gradients/lenet1/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
&lenet1/gradients/lenet1/Mean_grad/CastCast*lenet1/gradients/lenet1/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0
�
)lenet1/gradients/lenet1/Mean_grad/truedivRealDiv&lenet1/gradients/lenet1/Mean_grad/Tile&lenet1/gradients/lenet1/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
,lenet1/gradients/lenet1/Reshape_4_grad/ShapeShape$lenet1/SoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:*
out_type0
�
.lenet1/gradients/lenet1/Reshape_4_grad/ReshapeReshape)lenet1/gradients/lenet1/Mean_grad/truediv,lenet1/gradients/lenet1/Reshape_4_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
�
lenet1/gradients/zeros_like	ZerosLike&lenet1/SoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
Ilenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
Elenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims.lenet1/gradients/lenet1/Reshape_4_grad/ReshapeIlenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
>lenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/mulMulElenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims&lenet1/SoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
,lenet1/gradients/lenet1/Reshape_2_grad/ShapeShapelenet1/outputs/Sigmoid*
T0*
_output_shapes
:*
out_type0
�
.lenet1/gradients/lenet1/Reshape_2_grad/ReshapeReshape>lenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/mul,lenet1/gradients/lenet1/Reshape_2_grad/Shape*
T0*'
_output_shapes
:���������
*
Tshape0
�
8lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGradSigmoidGradlenet1/outputs/Sigmoid.lenet1/gradients/lenet1/Reshape_2_grad/Reshape*
T0*'
_output_shapes
:���������

�
8lenet1/gradients/lenet1/outputs/BiasAdd_grad/BiasAddGradBiasAddGrad8lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGrad*
T0*
_output_shapes
:
*
data_formatNHWC
�
=lenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/group_depsNoOp9^lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGrad9^lenet1/gradients/lenet1/outputs/BiasAdd_grad/BiasAddGrad
�
Elenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependencyIdentity8lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGrad>^lenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������

�
Glenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependency_1Identity8lenet1/gradients/lenet1/outputs/BiasAdd_grad/BiasAddGrad>^lenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@lenet1/gradients/lenet1/outputs/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

�
2lenet1/gradients/lenet1/outputs/MatMul_grad/MatMulMatMulElenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependencylenet1/outputs/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
4lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul_1MatMullenet1/Reshape_1Elenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�
*
transpose_a(*
transpose_b( 
�
<lenet1/gradients/lenet1/outputs/MatMul_grad/tuple/group_depsNoOp3^lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul5^lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul_1
�
Dlenet1/gradients/lenet1/outputs/MatMul_grad/tuple/control_dependencyIdentity2lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul=^lenet1/gradients/lenet1/outputs/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Flenet1/gradients/lenet1/outputs/MatMul_grad/tuple/control_dependency_1Identity4lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul_1=^lenet1/gradients/lenet1/outputs/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul_1*
_output_shapes
:	�

}
,lenet1/gradients/lenet1/Reshape_1_grad/ShapeShapelenet1/h4/AvgPool*
T0*
_output_shapes
:*
out_type0
�
.lenet1/gradients/lenet1/Reshape_1_grad/ReshapeReshapeDlenet1/gradients/lenet1/outputs/MatMul_grad/tuple/control_dependency,lenet1/gradients/lenet1/Reshape_1_grad/Shape*
T0*/
_output_shapes
:���������*
Tshape0
{
-lenet1/gradients/lenet1/h4/AvgPool_grad/ShapeShapelenet1/h3/Relu*
T0*
_output_shapes
:*
out_type0
�
3lenet1/gradients/lenet1/h4/AvgPool_grad/AvgPoolGradAvgPoolGrad-lenet1/gradients/lenet1/h4/AvgPool_grad/Shape.lenet1/gradients/lenet1/Reshape_1_grad/Reshape*/
_output_shapes
:���������*
paddingVALID*
ksize
*
T0*
data_formatNHWC*
strides

�
-lenet1/gradients/lenet1/h3/Relu_grad/ReluGradReluGrad3lenet1/gradients/lenet1/h4/AvgPool_grad/AvgPoolGradlenet1/h3/Relu*
T0*/
_output_shapes
:���������
�
3lenet1/gradients/lenet1/h3/BiasAdd_grad/BiasAddGradBiasAddGrad-lenet1/gradients/lenet1/h3/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
�
8lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/group_depsNoOp.^lenet1/gradients/lenet1/h3/Relu_grad/ReluGrad4^lenet1/gradients/lenet1/h3/BiasAdd_grad/BiasAddGrad
�
@lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependencyIdentity-lenet1/gradients/lenet1/h3/Relu_grad/ReluGrad9^lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@lenet1/gradients/lenet1/h3/Relu_grad/ReluGrad*/
_output_shapes
:���������
�
Blenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependency_1Identity3lenet1/gradients/lenet1/h3/BiasAdd_grad/BiasAddGrad9^lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@lenet1/gradients/lenet1/h3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
1lenet1/gradients/lenet1/h3/convolution_grad/ShapeShapelenet1/h2/AvgPool*
T0*
_output_shapes
:*
out_type0
�
?lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput1lenet1/gradients/lenet1/h3/convolution_grad/Shapelenet1/h3/kernel/read@lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides

�
3lenet1/gradients/lenet1/h3/convolution_grad/Shape_1Const*
_output_shapes
:*%
valueB"            *
dtype0
�
@lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterlenet1/h2/AvgPool3lenet1/gradients/lenet1/h3/convolution_grad/Shape_1@lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides

�
<lenet1/gradients/lenet1/h3/convolution_grad/tuple/group_depsNoOp@^lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropInputA^lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropFilter
�
Dlenet1/gradients/lenet1/h3/convolution_grad/tuple/control_dependencyIdentity?lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropInput=^lenet1/gradients/lenet1/h3/convolution_grad/tuple/group_deps*
T0*R
_classH
FDloc:@lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
Flenet1/gradients/lenet1/h3/convolution_grad/tuple/control_dependency_1Identity@lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropFilter=^lenet1/gradients/lenet1/h3/convolution_grad/tuple/group_deps*
T0*S
_classI
GEloc:@lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:
{
-lenet1/gradients/lenet1/h2/AvgPool_grad/ShapeShapelenet1/h1/Relu*
T0*
_output_shapes
:*
out_type0
�
3lenet1/gradients/lenet1/h2/AvgPool_grad/AvgPoolGradAvgPoolGrad-lenet1/gradients/lenet1/h2/AvgPool_grad/ShapeDlenet1/gradients/lenet1/h3/convolution_grad/tuple/control_dependency*/
_output_shapes
:���������*
paddingVALID*
ksize
*
T0*
data_formatNHWC*
strides

�
-lenet1/gradients/lenet1/h1/Relu_grad/ReluGradReluGrad3lenet1/gradients/lenet1/h2/AvgPool_grad/AvgPoolGradlenet1/h1/Relu*
T0*/
_output_shapes
:���������
�
3lenet1/gradients/lenet1/h1/BiasAdd_grad/BiasAddGradBiasAddGrad-lenet1/gradients/lenet1/h1/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
�
8lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/group_depsNoOp.^lenet1/gradients/lenet1/h1/Relu_grad/ReluGrad4^lenet1/gradients/lenet1/h1/BiasAdd_grad/BiasAddGrad
�
@lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependencyIdentity-lenet1/gradients/lenet1/h1/Relu_grad/ReluGrad9^lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@lenet1/gradients/lenet1/h1/Relu_grad/ReluGrad*/
_output_shapes
:���������
�
Blenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependency_1Identity3lenet1/gradients/lenet1/h1/BiasAdd_grad/BiasAddGrad9^lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@lenet1/gradients/lenet1/h1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

1lenet1/gradients/lenet1/h1/convolution_grad/ShapeShapelenet1/Reshape*
T0*
_output_shapes
:*
out_type0
�
?lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput1lenet1/gradients/lenet1/h1/convolution_grad/Shapelenet1/h1/kernel/read@lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides

�
3lenet1/gradients/lenet1/h1/convolution_grad/Shape_1Const*
_output_shapes
:*%
valueB"            *
dtype0
�
@lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterlenet1/Reshape3lenet1/gradients/lenet1/h1/convolution_grad/Shape_1@lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependency*&
_output_shapes
:*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides

�
<lenet1/gradients/lenet1/h1/convolution_grad/tuple/group_depsNoOp@^lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropInputA^lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropFilter
�
Dlenet1/gradients/lenet1/h1/convolution_grad/tuple/control_dependencyIdentity?lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropInput=^lenet1/gradients/lenet1/h1/convolution_grad/tuple/group_deps*
T0*R
_classH
FDloc:@lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
Flenet1/gradients/lenet1/h1/convolution_grad/tuple/control_dependency_1Identity@lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropFilter=^lenet1/gradients/lenet1/h1/convolution_grad/tuple/group_deps*
T0*S
_classI
GEloc:@lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
 lenet1/beta1_power/initial_valueConst*#
_class
loc:@lenet1/h1/kernel*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
lenet1/beta1_power
VariableV2*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 
�
lenet1/beta1_power/AssignAssignlenet1/beta1_power lenet1/beta1_power/initial_value*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking(
}
lenet1/beta1_power/readIdentitylenet1/beta1_power*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
 lenet1/beta2_power/initial_valueConst*#
_class
loc:@lenet1/h1/kernel*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
lenet1/beta2_power
VariableV2*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 
�
lenet1/beta2_power/AssignAssignlenet1/beta2_power lenet1/beta2_power/initial_value*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking(
}
lenet1/beta2_power/readIdentitylenet1/beta2_power*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
.lenet1/lenet1/h1/kernel/Adam/Initializer/ConstConst*#
_class
loc:@lenet1/h1/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
�
lenet1/lenet1/h1/kernel/Adam
VariableV2*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
#lenet1/lenet1/h1/kernel/Adam/AssignAssignlenet1/lenet1/h1/kernel/Adam.lenet1/lenet1/h1/kernel/Adam/Initializer/Const*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
!lenet1/lenet1/h1/kernel/Adam/readIdentitylenet1/lenet1/h1/kernel/Adam*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
0lenet1/lenet1/h1/kernel/Adam_1/Initializer/ConstConst*#
_class
loc:@lenet1/h1/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
�
lenet1/lenet1/h1/kernel/Adam_1
VariableV2*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
%lenet1/lenet1/h1/kernel/Adam_1/AssignAssignlenet1/lenet1/h1/kernel/Adam_10lenet1/lenet1/h1/kernel/Adam_1/Initializer/Const*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
#lenet1/lenet1/h1/kernel/Adam_1/readIdentitylenet1/lenet1/h1/kernel/Adam_1*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
,lenet1/lenet1/h1/bias/Adam/Initializer/ConstConst*!
_class
loc:@lenet1/h1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/lenet1/h1/bias/Adam
VariableV2*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
!lenet1/lenet1/h1/bias/Adam/AssignAssignlenet1/lenet1/h1/bias/Adam,lenet1/lenet1/h1/bias/Adam/Initializer/Const*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
lenet1/lenet1/h1/bias/Adam/readIdentitylenet1/lenet1/h1/bias/Adam*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:
�
.lenet1/lenet1/h1/bias/Adam_1/Initializer/ConstConst*!
_class
loc:@lenet1/h1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/lenet1/h1/bias/Adam_1
VariableV2*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
#lenet1/lenet1/h1/bias/Adam_1/AssignAssignlenet1/lenet1/h1/bias/Adam_1.lenet1/lenet1/h1/bias/Adam_1/Initializer/Const*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
!lenet1/lenet1/h1/bias/Adam_1/readIdentitylenet1/lenet1/h1/bias/Adam_1*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:
�
.lenet1/lenet1/h3/kernel/Adam/Initializer/ConstConst*#
_class
loc:@lenet1/h3/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
�
lenet1/lenet1/h3/kernel/Adam
VariableV2*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
#lenet1/lenet1/h3/kernel/Adam/AssignAssignlenet1/lenet1/h3/kernel/Adam.lenet1/lenet1/h3/kernel/Adam/Initializer/Const*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
!lenet1/lenet1/h3/kernel/Adam/readIdentitylenet1/lenet1/h3/kernel/Adam*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
0lenet1/lenet1/h3/kernel/Adam_1/Initializer/ConstConst*#
_class
loc:@lenet1/h3/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
�
lenet1/lenet1/h3/kernel/Adam_1
VariableV2*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
%lenet1/lenet1/h3/kernel/Adam_1/AssignAssignlenet1/lenet1/h3/kernel/Adam_10lenet1/lenet1/h3/kernel/Adam_1/Initializer/Const*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
#lenet1/lenet1/h3/kernel/Adam_1/readIdentitylenet1/lenet1/h3/kernel/Adam_1*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
,lenet1/lenet1/h3/bias/Adam/Initializer/ConstConst*!
_class
loc:@lenet1/h3/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/lenet1/h3/bias/Adam
VariableV2*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
!lenet1/lenet1/h3/bias/Adam/AssignAssignlenet1/lenet1/h3/bias/Adam,lenet1/lenet1/h3/bias/Adam/Initializer/Const*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
lenet1/lenet1/h3/bias/Adam/readIdentitylenet1/lenet1/h3/bias/Adam*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:
�
.lenet1/lenet1/h3/bias/Adam_1/Initializer/ConstConst*!
_class
loc:@lenet1/h3/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/lenet1/h3/bias/Adam_1
VariableV2*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
#lenet1/lenet1/h3/bias/Adam_1/AssignAssignlenet1/lenet1/h3/bias/Adam_1.lenet1/lenet1/h3/bias/Adam_1/Initializer/Const*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
!lenet1/lenet1/h3/bias/Adam_1/readIdentitylenet1/lenet1/h3/bias/Adam_1*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:
�
3lenet1/lenet1/outputs/kernel/Adam/Initializer/ConstConst*(
_class
loc:@lenet1/outputs/kernel*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
!lenet1/lenet1/outputs/kernel/Adam
VariableV2*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
shape:	�
*
	container *
dtype0*
shared_name 
�
(lenet1/lenet1/outputs/kernel/Adam/AssignAssign!lenet1/lenet1/outputs/kernel/Adam3lenet1/lenet1/outputs/kernel/Adam/Initializer/Const*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
&lenet1/lenet1/outputs/kernel/Adam/readIdentity!lenet1/lenet1/outputs/kernel/Adam*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
5lenet1/lenet1/outputs/kernel/Adam_1/Initializer/ConstConst*(
_class
loc:@lenet1/outputs/kernel*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
#lenet1/lenet1/outputs/kernel/Adam_1
VariableV2*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
shape:	�
*
	container *
dtype0*
shared_name 
�
*lenet1/lenet1/outputs/kernel/Adam_1/AssignAssign#lenet1/lenet1/outputs/kernel/Adam_15lenet1/lenet1/outputs/kernel/Adam_1/Initializer/Const*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
(lenet1/lenet1/outputs/kernel/Adam_1/readIdentity#lenet1/lenet1/outputs/kernel/Adam_1*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
1lenet1/lenet1/outputs/bias/Adam/Initializer/ConstConst*&
_class
loc:@lenet1/outputs/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
lenet1/lenet1/outputs/bias/Adam
VariableV2*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
shape:
*
	container *
dtype0*
shared_name 
�
&lenet1/lenet1/outputs/bias/Adam/AssignAssignlenet1/lenet1/outputs/bias/Adam1lenet1/lenet1/outputs/bias/Adam/Initializer/Const*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
$lenet1/lenet1/outputs/bias/Adam/readIdentitylenet1/lenet1/outputs/bias/Adam*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:

�
3lenet1/lenet1/outputs/bias/Adam_1/Initializer/ConstConst*&
_class
loc:@lenet1/outputs/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
!lenet1/lenet1/outputs/bias/Adam_1
VariableV2*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
shape:
*
	container *
dtype0*
shared_name 
�
(lenet1/lenet1/outputs/bias/Adam_1/AssignAssign!lenet1/lenet1/outputs/bias/Adam_13lenet1/lenet1/outputs/bias/Adam_1/Initializer/Const*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
&lenet1/lenet1/outputs/bias/Adam_1/readIdentity!lenet1/lenet1/outputs/bias/Adam_1*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:

^
lenet1/Adam/learning_rateConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
V
lenet1/Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
V
lenet1/Adam/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
X
lenet1/Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
-lenet1/Adam/update_lenet1/h1/kernel/ApplyAdam	ApplyAdamlenet1/h1/kernellenet1/lenet1/h1/kernel/Adamlenet1/lenet1/h1/kernel/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonFlenet1/gradients/lenet1/h1/convolution_grad/tuple/control_dependency_1*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
use_locking( 
�
+lenet1/Adam/update_lenet1/h1/bias/ApplyAdam	ApplyAdamlenet1/h1/biaslenet1/lenet1/h1/bias/Adamlenet1/lenet1/h1/bias/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonBlenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependency_1*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
use_locking( 
�
-lenet1/Adam/update_lenet1/h3/kernel/ApplyAdam	ApplyAdamlenet1/h3/kernellenet1/lenet1/h3/kernel/Adamlenet1/lenet1/h3/kernel/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonFlenet1/gradients/lenet1/h3/convolution_grad/tuple/control_dependency_1*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
use_locking( 
�
+lenet1/Adam/update_lenet1/h3/bias/ApplyAdam	ApplyAdamlenet1/h3/biaslenet1/lenet1/h3/bias/Adamlenet1/lenet1/h3/bias/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonBlenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependency_1*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
use_locking( 
�
2lenet1/Adam/update_lenet1/outputs/kernel/ApplyAdam	ApplyAdamlenet1/outputs/kernel!lenet1/lenet1/outputs/kernel/Adam#lenet1/lenet1/outputs/kernel/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonFlenet1/gradients/lenet1/outputs/MatMul_grad/tuple/control_dependency_1*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
use_locking( 
�
0lenet1/Adam/update_lenet1/outputs/bias/ApplyAdam	ApplyAdamlenet1/outputs/biaslenet1/lenet1/outputs/bias/Adam!lenet1/lenet1/outputs/bias/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonGlenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependency_1*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
use_locking( 
�
lenet1/Adam/mulMullenet1/beta1_power/readlenet1/Adam/beta1.^lenet1/Adam/update_lenet1/h1/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h1/bias/ApplyAdam.^lenet1/Adam/update_lenet1/h3/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h3/bias/ApplyAdam3^lenet1/Adam/update_lenet1/outputs/kernel/ApplyAdam1^lenet1/Adam/update_lenet1/outputs/bias/ApplyAdam*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
lenet1/Adam/AssignAssignlenet1/beta1_powerlenet1/Adam/mul*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking( 
�
lenet1/Adam/mul_1Mullenet1/beta2_power/readlenet1/Adam/beta2.^lenet1/Adam/update_lenet1/h1/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h1/bias/ApplyAdam.^lenet1/Adam/update_lenet1/h3/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h3/bias/ApplyAdam3^lenet1/Adam/update_lenet1/outputs/kernel/ApplyAdam1^lenet1/Adam/update_lenet1/outputs/bias/ApplyAdam*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
lenet1/Adam/Assign_1Assignlenet1/beta2_powerlenet1/Adam/mul_1*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking( 
�
lenet1/Adam/updateNoOp.^lenet1/Adam/update_lenet1/h1/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h1/bias/ApplyAdam.^lenet1/Adam/update_lenet1/h3/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h3/bias/ApplyAdam3^lenet1/Adam/update_lenet1/outputs/kernel/ApplyAdam1^lenet1/Adam/update_lenet1/outputs/bias/ApplyAdam^lenet1/Adam/Assign^lenet1/Adam/Assign_1
�
lenet1/Adam/valueConst^lenet1/Adam/update*%
_class
loc:@lenet1/global_step*
value	B	 R*
dtype0	*
_output_shapes
: 
�
lenet1/Adam	AssignAddlenet1/global_steplenet1/Adam/value*
T0	*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
use_locking( 
Y
lenet1/ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
�
lenet1/ArgMaxArgMaxlenet1/outputs/Sigmoidlenet1/ArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:���������
[
lenet1/ArgMax_1/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
}
lenet1/ArgMax_1ArgMaxlenet1/labelslenet1/ArgMax_1/dimension*
T0*

Tidx0*#
_output_shapes
:���������
c
lenet1/EqualEquallenet1/ArgMaxlenet1/ArgMax_1*
T0	*#
_output_shapes
:���������
`
lenet1/Cast_1Castlenet1/Equal*#
_output_shapes
:���������*

SrcT0
*

DstT0
X
lenet1/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
t
lenet1/accuracyMeanlenet1/Cast_1lenet1/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
p
lenet1/train_accuracy/tagsConst*
_output_shapes
: *&
valueB Blenet1/train_accuracy*
dtype0
t
lenet1/train_accuracyScalarSummarylenet1/train_accuracy/tagslenet1/accuracy*
T0*
_output_shapes
: 
n
lenet1/Merge/MergeSummaryMergeSummarylenet1/losslenet1/train_accuracy*
_output_shapes
: *
N
�
initNoOp^lenet1/h1/kernel/Assign^lenet1/h1/bias/Assign^lenet1/h3/kernel/Assign^lenet1/h3/bias/Assign^lenet1/outputs/kernel/Assign^lenet1/outputs/bias/Assign^lenet1/global_step/Assign^lenet1/beta1_power/Assign^lenet1/beta2_power/Assign$^lenet1/lenet1/h1/kernel/Adam/Assign&^lenet1/lenet1/h1/kernel/Adam_1/Assign"^lenet1/lenet1/h1/bias/Adam/Assign$^lenet1/lenet1/h1/bias/Adam_1/Assign$^lenet1/lenet1/h3/kernel/Adam/Assign&^lenet1/lenet1/h3/kernel/Adam_1/Assign"^lenet1/lenet1/h3/bias/Adam/Assign$^lenet1/lenet1/h3/bias/Adam_1/Assign)^lenet1/lenet1/outputs/kernel/Adam/Assign+^lenet1/lenet1/outputs/kernel/Adam_1/Assign'^lenet1/lenet1/outputs/bias/Adam/Assign)^lenet1/lenet1/outputs/bias/Adam_1/Assign
P

save/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0
�
save/SaveV2/tensor_namesConst*
_output_shapes
:*�
value�B�Blenet1/beta1_powerBlenet1/beta2_powerBlenet1/global_stepBlenet1/h1/biasBlenet1/h1/kernelBlenet1/h3/biasBlenet1/h3/kernelBlenet1/lenet1/h1/bias/AdamBlenet1/lenet1/h1/bias/Adam_1Blenet1/lenet1/h1/kernel/AdamBlenet1/lenet1/h1/kernel/Adam_1Blenet1/lenet1/h3/bias/AdamBlenet1/lenet1/h3/bias/Adam_1Blenet1/lenet1/h3/kernel/AdamBlenet1/lenet1/h3/kernel/Adam_1Blenet1/lenet1/outputs/bias/AdamB!lenet1/lenet1/outputs/bias/Adam_1B!lenet1/lenet1/outputs/kernel/AdamB#lenet1/lenet1/outputs/kernel/Adam_1Blenet1/outputs/biasBlenet1/outputs/kernel*
dtype0
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslenet1/beta1_powerlenet1/beta2_powerlenet1/global_steplenet1/h1/biaslenet1/h1/kernellenet1/h3/biaslenet1/h3/kernellenet1/lenet1/h1/bias/Adamlenet1/lenet1/h1/bias/Adam_1lenet1/lenet1/h1/kernel/Adamlenet1/lenet1/h1/kernel/Adam_1lenet1/lenet1/h3/bias/Adamlenet1/lenet1/h3/bias/Adam_1lenet1/lenet1/h3/kernel/Adamlenet1/lenet1/h3/kernel/Adam_1lenet1/lenet1/outputs/bias/Adam!lenet1/lenet1/outputs/bias/Adam_1!lenet1/lenet1/outputs/kernel/Adam#lenet1/lenet1/outputs/kernel/Adam_1lenet1/outputs/biaslenet1/outputs/kernel*#
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
v
save/RestoreV2/tensor_namesConst*
_output_shapes
:*'
valueBBlenet1/beta1_power*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignlenet1/beta1_powersave/RestoreV2*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking(
x
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:*'
valueBBlenet1/beta2_power*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assignlenet1/beta2_powersave/RestoreV2_1*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking(
x
save/RestoreV2_2/tensor_namesConst*
_output_shapes
:*'
valueBBlenet1/global_step*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2	*
_output_shapes
:
�
save/Assign_2Assignlenet1/global_stepsave/RestoreV2_2*
T0	*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
validate_shape(*
use_locking(
t
save/RestoreV2_3/tensor_namesConst*
_output_shapes
:*#
valueBBlenet1/h1/bias*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignlenet1/h1/biassave/RestoreV2_3*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
v
save/RestoreV2_4/tensor_namesConst*
_output_shapes
:*%
valueBBlenet1/h1/kernel*
dtype0
j
!save/RestoreV2_4/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assignlenet1/h1/kernelsave/RestoreV2_4*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
t
save/RestoreV2_5/tensor_namesConst*
_output_shapes
:*#
valueBBlenet1/h3/bias*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assignlenet1/h3/biassave/RestoreV2_5*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
v
save/RestoreV2_6/tensor_namesConst*
_output_shapes
:*%
valueBBlenet1/h3/kernel*
dtype0
j
!save/RestoreV2_6/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assignlenet1/h3/kernelsave/RestoreV2_6*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_7/tensor_namesConst*
_output_shapes
:*/
value&B$Blenet1/lenet1/h1/bias/Adam*
dtype0
j
!save/RestoreV2_7/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assignlenet1/lenet1/h1/bias/Adamsave/RestoreV2_7*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_8/tensor_namesConst*
_output_shapes
:*1
value(B&Blenet1/lenet1/h1/bias/Adam_1*
dtype0
j
!save/RestoreV2_8/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assignlenet1/lenet1/h1/bias/Adam_1save/RestoreV2_8*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_9/tensor_namesConst*
_output_shapes
:*1
value(B&Blenet1/lenet1/h1/kernel/Adam*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assignlenet1/lenet1/h1/kernel/Adamsave/RestoreV2_9*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_10/tensor_namesConst*
_output_shapes
:*3
value*B(Blenet1/lenet1/h1/kernel/Adam_1*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assignlenet1/lenet1/h1/kernel/Adam_1save/RestoreV2_10*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_11/tensor_namesConst*
_output_shapes
:*/
value&B$Blenet1/lenet1/h3/bias/Adam*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assignlenet1/lenet1/h3/bias/Adamsave/RestoreV2_11*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_12/tensor_namesConst*
_output_shapes
:*1
value(B&Blenet1/lenet1/h3/bias/Adam_1*
dtype0
k
"save/RestoreV2_12/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assignlenet1/lenet1/h3/bias/Adam_1save/RestoreV2_12*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_13/tensor_namesConst*
_output_shapes
:*1
value(B&Blenet1/lenet1/h3/kernel/Adam*
dtype0
k
"save/RestoreV2_13/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assignlenet1/lenet1/h3/kernel/Adamsave/RestoreV2_13*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_14/tensor_namesConst*
_output_shapes
:*3
value*B(Blenet1/lenet1/h3/kernel/Adam_1*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assignlenet1/lenet1/h3/kernel/Adam_1save/RestoreV2_14*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_15/tensor_namesConst*
_output_shapes
:*4
value+B)Blenet1/lenet1/outputs/bias/Adam*
dtype0
k
"save/RestoreV2_15/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15Assignlenet1/lenet1/outputs/bias/Adamsave/RestoreV2_15*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
save/RestoreV2_16/tensor_namesConst*
_output_shapes
:*6
value-B+B!lenet1/lenet1/outputs/bias/Adam_1*
dtype0
k
"save/RestoreV2_16/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assign!lenet1/lenet1/outputs/bias/Adam_1save/RestoreV2_16*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
save/RestoreV2_17/tensor_namesConst*
_output_shapes
:*6
value-B+B!lenet1/lenet1/outputs/kernel/Adam*
dtype0
k
"save/RestoreV2_17/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assign!lenet1/lenet1/outputs/kernel/Adamsave/RestoreV2_17*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
save/RestoreV2_18/tensor_namesConst*
_output_shapes
:*8
value/B-B#lenet1/lenet1/outputs/kernel/Adam_1*
dtype0
k
"save/RestoreV2_18/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assign#lenet1/lenet1/outputs/kernel/Adam_1save/RestoreV2_18*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
z
save/RestoreV2_19/tensor_namesConst*
_output_shapes
:*(
valueBBlenet1/outputs/bias*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assignlenet1/outputs/biassave/RestoreV2_19*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
|
save/RestoreV2_20/tensor_namesConst*
_output_shapes
:**
value!BBlenet1/outputs/kernel*
dtype0
k
"save/RestoreV2_20/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assignlenet1/outputs/kernelsave/RestoreV2_20*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedlenet1/h1/kernel*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedlenet1/h1/bias*!
_class
loc:@lenet1/h1/bias*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedlenet1/h3/kernel*#
_class
loc:@lenet1/h3/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedlenet1/h3/bias*!
_class
loc:@lenet1/h3/bias*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedlenet1/outputs/kernel*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedlenet1/outputs/bias*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedlenet1/global_step*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
dtype0	
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedlenet1/beta1_power*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializedlenet1/beta2_power*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedlenet1/lenet1/h1/kernel/Adam*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedlenet1/lenet1/h1/kernel/Adam_1*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedlenet1/lenet1/h1/bias/Adam*!
_class
loc:@lenet1/h1/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedlenet1/lenet1/h1/bias/Adam_1*!
_class
loc:@lenet1/h1/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedlenet1/lenet1/h3/kernel/Adam*#
_class
loc:@lenet1/h3/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedlenet1/lenet1/h3/kernel/Adam_1*#
_class
loc:@lenet1/h3/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedlenet1/lenet1/h3/bias/Adam*!
_class
loc:@lenet1/h3/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedlenet1/lenet1/h3/bias/Adam_1*!
_class
loc:@lenet1/h3/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized!lenet1/lenet1/outputs/kernel/Adam*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized#lenet1/lenet1/outputs/kernel/Adam_1*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedlenet1/lenet1/outputs/bias/Adam*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized!lenet1/lenet1/outputs/bias/Adam_1*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
: *
dtype0
�

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_20*
T0
*

axis *
_output_shapes
:*
N
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst*
_output_shapes
:*�
value�B�Blenet1/h1/kernelBlenet1/h1/biasBlenet1/h3/kernelBlenet1/h3/biasBlenet1/outputs/kernelBlenet1/outputs/biasBlenet1/global_stepBlenet1/beta1_powerBlenet1/beta2_powerBlenet1/lenet1/h1/kernel/AdamBlenet1/lenet1/h1/kernel/Adam_1Blenet1/lenet1/h1/bias/AdamBlenet1/lenet1/h1/bias/Adam_1Blenet1/lenet1/h3/kernel/AdamBlenet1/lenet1/h3/kernel/Adam_1Blenet1/lenet1/h3/bias/AdamBlenet1/lenet1/h3/bias/Adam_1B!lenet1/lenet1/outputs/kernel/AdamB#lenet1/lenet1/outputs/kernel/Adam_1Blenet1/lenet1/outputs/bias/AdamB!lenet1/lenet1/outputs/bias/Adam_1*
dtype0
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
ellipsis_mask *

begin_mask*
_output_shapes
:*
end_mask *
new_axis_mask *
T0*
Index0*
shrink_axis_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
valueB: *
dtype0
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
valueB:*
dtype0
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask *
_output_shapes
: *
end_mask*
new_axis_mask *
T0*
Index0*
shrink_axis_mask 
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
T0*

axis *
_output_shapes
:*
N
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
T0*
_output_shapes
:*
Tshape0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
valueB:
���������*
dtype0
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
_output_shapes
:*
Tshape0
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
T0	*
squeeze_dims
*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*#
_output_shapes
:���������*
Tparams0*
validate_indices(*
Tindices0	

init_1NoOp

init_all_tablesNoOp
-

group_depsNoOp^init_1^init_all_tables"w��	�6     l��	,,&����AJ��
�)�)
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
�
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
�
AvgPoolGrad
orig_input_shape	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
�
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�


LogicalNot
x

y

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
/
Sigmoid
x"T
y"T"
Ttype:	
2
;
SigmoidGrad
x"T
y"T
z"T"
Ttype:	
2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.1.02v1.1.0-rc0-61-g1ec6ed5��
`
lenet1/inputsPlaceholder*(
_output_shapes
:����������*
shape: *
dtype0
m
lenet1/Reshape/shapeConst*%
valueB"����         *
_output_shapes
:*
dtype0
�
lenet1/ReshapeReshapelenet1/inputslenet1/Reshape/shape*
T0*/
_output_shapes
:���������*
Tshape0
_
lenet1/labelsPlaceholder*'
_output_shapes
:���������
*
shape: *
dtype0
�
1lenet1/h1/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@lenet1/h1/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
/lenet1/h1/kernel/Initializer/random_uniform/minConst*#
_class
loc:@lenet1/h1/kernel*
valueB
 *�X`�*
dtype0*
_output_shapes
: 
�
/lenet1/h1/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@lenet1/h1/kernel*
valueB
 *�X`>*
dtype0*
_output_shapes
: 
�
9lenet1/h1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1lenet1/h1/kernel/Initializer/random_uniform/shape*#
_class
loc:@lenet1/h1/kernel*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:
�
/lenet1/h1/kernel/Initializer/random_uniform/subSub/lenet1/h1/kernel/Initializer/random_uniform/max/lenet1/h1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
/lenet1/h1/kernel/Initializer/random_uniform/mulMul9lenet1/h1/kernel/Initializer/random_uniform/RandomUniform/lenet1/h1/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
+lenet1/h1/kernel/Initializer/random_uniformAdd/lenet1/h1/kernel/Initializer/random_uniform/mul/lenet1/h1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
lenet1/h1/kernel
VariableV2*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
lenet1/h1/kernel/AssignAssignlenet1/h1/kernel+lenet1/h1/kernel/Initializer/random_uniform*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
lenet1/h1/kernel/readIdentitylenet1/h1/kernel*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
 lenet1/h1/bias/Initializer/ConstConst*!
_class
loc:@lenet1/h1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/h1/bias
VariableV2*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
lenet1/h1/bias/AssignAssignlenet1/h1/bias lenet1/h1/bias/Initializer/Const*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
w
lenet1/h1/bias/readIdentitylenet1/h1/bias*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:
t
lenet1/h1/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
t
#lenet1/h1/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
lenet1/h1/convolutionConv2Dlenet1/Reshapelenet1/h1/kernel/read*
strides
*/
_output_shapes
:���������*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC
�
lenet1/h1/BiasAddBiasAddlenet1/h1/convolutionlenet1/h1/bias/read*
T0*/
_output_shapes
:���������*
data_formatNHWC
c
lenet1/h1/ReluRelulenet1/h1/BiasAdd*
T0*/
_output_shapes
:���������
�
lenet1/h2/AvgPoolAvgPoollenet1/h1/Relu*/
_output_shapes
:���������*
paddingVALID*
ksize
*
T0*
strides
*
data_formatNHWC
�
1lenet1/h3/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@lenet1/h3/kernel*%
valueB"            *
dtype0*
_output_shapes
:
�
/lenet1/h3/kernel/Initializer/random_uniform/minConst*#
_class
loc:@lenet1/h3/kernel*
valueB
 *����*
dtype0*
_output_shapes
: 
�
/lenet1/h3/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@lenet1/h3/kernel*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
9lenet1/h3/kernel/Initializer/random_uniform/RandomUniformRandomUniform1lenet1/h3/kernel/Initializer/random_uniform/shape*#
_class
loc:@lenet1/h3/kernel*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:
�
/lenet1/h3/kernel/Initializer/random_uniform/subSub/lenet1/h3/kernel/Initializer/random_uniform/max/lenet1/h3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@lenet1/h3/kernel*
_output_shapes
: 
�
/lenet1/h3/kernel/Initializer/random_uniform/mulMul9lenet1/h3/kernel/Initializer/random_uniform/RandomUniform/lenet1/h3/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
+lenet1/h3/kernel/Initializer/random_uniformAdd/lenet1/h3/kernel/Initializer/random_uniform/mul/lenet1/h3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
lenet1/h3/kernel
VariableV2*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
lenet1/h3/kernel/AssignAssignlenet1/h3/kernel+lenet1/h3/kernel/Initializer/random_uniform*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
lenet1/h3/kernel/readIdentitylenet1/h3/kernel*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
 lenet1/h3/bias/Initializer/ConstConst*!
_class
loc:@lenet1/h3/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/h3/bias
VariableV2*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
lenet1/h3/bias/AssignAssignlenet1/h3/bias lenet1/h3/bias/Initializer/Const*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
w
lenet1/h3/bias/readIdentitylenet1/h3/bias*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:
t
lenet1/h3/convolution/ShapeConst*%
valueB"            *
_output_shapes
:*
dtype0
t
#lenet1/h3/convolution/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
�
lenet1/h3/convolutionConv2Dlenet1/h2/AvgPoollenet1/h3/kernel/read*
strides
*/
_output_shapes
:���������*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC
�
lenet1/h3/BiasAddBiasAddlenet1/h3/convolutionlenet1/h3/bias/read*
T0*/
_output_shapes
:���������*
data_formatNHWC
c
lenet1/h3/ReluRelulenet1/h3/BiasAdd*
T0*/
_output_shapes
:���������
�
lenet1/h4/AvgPoolAvgPoollenet1/h3/Relu*/
_output_shapes
:���������*
paddingVALID*
ksize
*
T0*
strides
*
data_formatNHWC
g
lenet1/Reshape_1/shapeConst*
valueB"�����   *
_output_shapes
:*
dtype0
�
lenet1/Reshape_1Reshapelenet1/h4/AvgPoollenet1/Reshape_1/shape*
T0*(
_output_shapes
:����������*
Tshape0
�
6lenet1/outputs/kernel/Initializer/random_uniform/shapeConst*(
_class
loc:@lenet1/outputs/kernel*
valueB"�   
   *
dtype0*
_output_shapes
:
�
4lenet1/outputs/kernel/Initializer/random_uniform/minConst*(
_class
loc:@lenet1/outputs/kernel*
valueB
 *W{0�*
dtype0*
_output_shapes
: 
�
4lenet1/outputs/kernel/Initializer/random_uniform/maxConst*(
_class
loc:@lenet1/outputs/kernel*
valueB
 *W{0>*
dtype0*
_output_shapes
: 
�
>lenet1/outputs/kernel/Initializer/random_uniform/RandomUniformRandomUniform6lenet1/outputs/kernel/Initializer/random_uniform/shape*(
_class
loc:@lenet1/outputs/kernel*

seed *
seed2 *
dtype0*
T0*
_output_shapes
:	�

�
4lenet1/outputs/kernel/Initializer/random_uniform/subSub4lenet1/outputs/kernel/Initializer/random_uniform/max4lenet1/outputs/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
: 
�
4lenet1/outputs/kernel/Initializer/random_uniform/mulMul>lenet1/outputs/kernel/Initializer/random_uniform/RandomUniform4lenet1/outputs/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
0lenet1/outputs/kernel/Initializer/random_uniformAdd4lenet1/outputs/kernel/Initializer/random_uniform/mul4lenet1/outputs/kernel/Initializer/random_uniform/min*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
lenet1/outputs/kernel
VariableV2*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
shape:	�
*
	container *
dtype0*
shared_name 
�
lenet1/outputs/kernel/AssignAssignlenet1/outputs/kernel0lenet1/outputs/kernel/Initializer/random_uniform*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
lenet1/outputs/kernel/readIdentitylenet1/outputs/kernel*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
%lenet1/outputs/bias/Initializer/ConstConst*&
_class
loc:@lenet1/outputs/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
lenet1/outputs/bias
VariableV2*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
shape:
*
	container *
dtype0*
shared_name 
�
lenet1/outputs/bias/AssignAssignlenet1/outputs/bias%lenet1/outputs/bias/Initializer/Const*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
lenet1/outputs/bias/readIdentitylenet1/outputs/bias*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:

�
lenet1/outputs/MatMulMatMullenet1/Reshape_1lenet1/outputs/kernel/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
lenet1/outputs/BiasAddBiasAddlenet1/outputs/MatMullenet1/outputs/bias/read*
T0*'
_output_shapes
:���������
*
data_formatNHWC
k
lenet1/outputs/SigmoidSigmoidlenet1/outputs/BiasAdd*
T0*'
_output_shapes
:���������

�
$lenet1/global_step/Initializer/ConstConst*%
_class
loc:@lenet1/global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
�
lenet1/global_step
VariableV2*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
shape: *
	container *
dtype0	*
shared_name 
�
lenet1/global_step/AssignAssignlenet1/global_step$lenet1/global_step/Initializer/Const*
T0	*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
validate_shape(*
use_locking(

lenet1/global_step/readIdentitylenet1/global_step*
T0	*%
_class
loc:@lenet1/global_step*
_output_shapes
: 
M
lenet1/RankConst*
value	B :*
_output_shapes
: *
dtype0
b
lenet1/ShapeShapelenet1/outputs/Sigmoid*
T0*
_output_shapes
:*
out_type0
O
lenet1/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
d
lenet1/Shape_1Shapelenet1/outputs/Sigmoid*
T0*
_output_shapes
:*
out_type0
N
lenet1/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
O

lenet1/SubSublenet1/Rank_1lenet1/Sub/y*
T0*
_output_shapes
: 
`
lenet1/Slice/beginPack
lenet1/Sub*
T0*

axis *
_output_shapes
:*
N
[
lenet1/Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
~
lenet1/SliceSlicelenet1/Shape_1lenet1/Slice/beginlenet1/Slice/size*
T0*
_output_shapes
:*
Index0
i
lenet1/concat/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
T
lenet1/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
lenet1/concatConcatV2lenet1/concat/values_0lenet1/Slicelenet1/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
�
lenet1/Reshape_2Reshapelenet1/outputs/Sigmoidlenet1/concat*
T0*0
_output_shapes
:������������������*
Tshape0
O
lenet1/Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
[
lenet1/Shape_2Shapelenet1/labels*
T0*
_output_shapes
:*
out_type0
P
lenet1/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
S
lenet1/Sub_1Sublenet1/Rank_2lenet1/Sub_1/y*
T0*
_output_shapes
: 
d
lenet1/Slice_1/beginPacklenet1/Sub_1*
T0*

axis *
_output_shapes
:*
N
]
lenet1/Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
�
lenet1/Slice_1Slicelenet1/Shape_2lenet1/Slice_1/beginlenet1/Slice_1/size*
T0*
_output_shapes
:*
Index0
k
lenet1/concat_1/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
V
lenet1/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
lenet1/concat_1ConcatV2lenet1/concat_1/values_0lenet1/Slice_1lenet1/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N
�
lenet1/Reshape_3Reshapelenet1/labelslenet1/concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
$lenet1/SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitslenet1/Reshape_2lenet1/Reshape_3*
T0*?
_output_shapes-
+:���������:������������������
P
lenet1/Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
Q
lenet1/Sub_2Sublenet1/Ranklenet1/Sub_2/y*
T0*
_output_shapes
: 
^
lenet1/Slice_2/beginConst*
valueB: *
_output_shapes
:*
dtype0
c
lenet1/Slice_2/sizePacklenet1/Sub_2*
T0*

axis *
_output_shapes
:*
N
�
lenet1/Slice_2Slicelenet1/Shapelenet1/Slice_2/beginlenet1/Slice_2/size*
T0*#
_output_shapes
:���������*
Index0
�
lenet1/Reshape_4Reshape$lenet1/SoftmaxCrossEntropyWithLogitslenet1/Slice_2*
T0*#
_output_shapes
:���������*
Tshape0
V
lenet1/ConstConst*
valueB: *
_output_shapes
:*
dtype0
q
lenet1/MeanMeanlenet1/Reshape_4lenet1/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
\
lenet1/loss/tagsConst*
valueB Blenet1/loss*
_output_shapes
: *
dtype0
\
lenet1/lossScalarSummarylenet1/loss/tagslenet1/Mean*
T0*
_output_shapes
: 
Y
lenet1/gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
[
lenet1/gradients/ConstConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
n
lenet1/gradients/FillFilllenet1/gradients/Shapelenet1/gradients/Const*
T0*
_output_shapes
: 
y
/lenet1/gradients/lenet1/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
)lenet1/gradients/lenet1/Mean_grad/ReshapeReshapelenet1/gradients/Fill/lenet1/gradients/lenet1/Mean_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
w
'lenet1/gradients/lenet1/Mean_grad/ShapeShapelenet1/Reshape_4*
T0*
_output_shapes
:*
out_type0
�
&lenet1/gradients/lenet1/Mean_grad/TileTile)lenet1/gradients/lenet1/Mean_grad/Reshape'lenet1/gradients/lenet1/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
y
)lenet1/gradients/lenet1/Mean_grad/Shape_1Shapelenet1/Reshape_4*
T0*
_output_shapes
:*
out_type0
l
)lenet1/gradients/lenet1/Mean_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
q
'lenet1/gradients/lenet1/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
&lenet1/gradients/lenet1/Mean_grad/ProdProd)lenet1/gradients/lenet1/Mean_grad/Shape_1'lenet1/gradients/lenet1/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
s
)lenet1/gradients/lenet1/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
�
(lenet1/gradients/lenet1/Mean_grad/Prod_1Prod)lenet1/gradients/lenet1/Mean_grad/Shape_2)lenet1/gradients/lenet1/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
m
+lenet1/gradients/lenet1/Mean_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
�
)lenet1/gradients/lenet1/Mean_grad/MaximumMaximum(lenet1/gradients/lenet1/Mean_grad/Prod_1+lenet1/gradients/lenet1/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
*lenet1/gradients/lenet1/Mean_grad/floordivFloorDiv&lenet1/gradients/lenet1/Mean_grad/Prod)lenet1/gradients/lenet1/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
&lenet1/gradients/lenet1/Mean_grad/CastCast*lenet1/gradients/lenet1/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0
�
)lenet1/gradients/lenet1/Mean_grad/truedivRealDiv&lenet1/gradients/lenet1/Mean_grad/Tile&lenet1/gradients/lenet1/Mean_grad/Cast*
T0*#
_output_shapes
:���������
�
,lenet1/gradients/lenet1/Reshape_4_grad/ShapeShape$lenet1/SoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:*
out_type0
�
.lenet1/gradients/lenet1/Reshape_4_grad/ReshapeReshape)lenet1/gradients/lenet1/Mean_grad/truediv,lenet1/gradients/lenet1/Reshape_4_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
�
lenet1/gradients/zeros_like	ZerosLike&lenet1/SoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
Ilenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
_output_shapes
: *
dtype0
�
Elenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims.lenet1/gradients/lenet1/Reshape_4_grad/ReshapeIlenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
�
>lenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/mulMulElenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/ExpandDims&lenet1/SoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
,lenet1/gradients/lenet1/Reshape_2_grad/ShapeShapelenet1/outputs/Sigmoid*
T0*
_output_shapes
:*
out_type0
�
.lenet1/gradients/lenet1/Reshape_2_grad/ReshapeReshape>lenet1/gradients/lenet1/SoftmaxCrossEntropyWithLogits_grad/mul,lenet1/gradients/lenet1/Reshape_2_grad/Shape*
T0*'
_output_shapes
:���������
*
Tshape0
�
8lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGradSigmoidGradlenet1/outputs/Sigmoid.lenet1/gradients/lenet1/Reshape_2_grad/Reshape*
T0*'
_output_shapes
:���������

�
8lenet1/gradients/lenet1/outputs/BiasAdd_grad/BiasAddGradBiasAddGrad8lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGrad*
T0*
_output_shapes
:
*
data_formatNHWC
�
=lenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/group_depsNoOp9^lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGrad9^lenet1/gradients/lenet1/outputs/BiasAdd_grad/BiasAddGrad
�
Elenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependencyIdentity8lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGrad>^lenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@lenet1/gradients/lenet1/outputs/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������

�
Glenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependency_1Identity8lenet1/gradients/lenet1/outputs/BiasAdd_grad/BiasAddGrad>^lenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@lenet1/gradients/lenet1/outputs/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

�
2lenet1/gradients/lenet1/outputs/MatMul_grad/MatMulMatMulElenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependencylenet1/outputs/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
4lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul_1MatMullenet1/Reshape_1Elenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�
*
transpose_a(*
transpose_b( 
�
<lenet1/gradients/lenet1/outputs/MatMul_grad/tuple/group_depsNoOp3^lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul5^lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul_1
�
Dlenet1/gradients/lenet1/outputs/MatMul_grad/tuple/control_dependencyIdentity2lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul=^lenet1/gradients/lenet1/outputs/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Flenet1/gradients/lenet1/outputs/MatMul_grad/tuple/control_dependency_1Identity4lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul_1=^lenet1/gradients/lenet1/outputs/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@lenet1/gradients/lenet1/outputs/MatMul_grad/MatMul_1*
_output_shapes
:	�

}
,lenet1/gradients/lenet1/Reshape_1_grad/ShapeShapelenet1/h4/AvgPool*
T0*
_output_shapes
:*
out_type0
�
.lenet1/gradients/lenet1/Reshape_1_grad/ReshapeReshapeDlenet1/gradients/lenet1/outputs/MatMul_grad/tuple/control_dependency,lenet1/gradients/lenet1/Reshape_1_grad/Shape*
T0*/
_output_shapes
:���������*
Tshape0
{
-lenet1/gradients/lenet1/h4/AvgPool_grad/ShapeShapelenet1/h3/Relu*
T0*
_output_shapes
:*
out_type0
�
3lenet1/gradients/lenet1/h4/AvgPool_grad/AvgPoolGradAvgPoolGrad-lenet1/gradients/lenet1/h4/AvgPool_grad/Shape.lenet1/gradients/lenet1/Reshape_1_grad/Reshape*/
_output_shapes
:���������*
paddingVALID*
ksize
*
T0*
strides
*
data_formatNHWC
�
-lenet1/gradients/lenet1/h3/Relu_grad/ReluGradReluGrad3lenet1/gradients/lenet1/h4/AvgPool_grad/AvgPoolGradlenet1/h3/Relu*
T0*/
_output_shapes
:���������
�
3lenet1/gradients/lenet1/h3/BiasAdd_grad/BiasAddGradBiasAddGrad-lenet1/gradients/lenet1/h3/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
�
8lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/group_depsNoOp.^lenet1/gradients/lenet1/h3/Relu_grad/ReluGrad4^lenet1/gradients/lenet1/h3/BiasAdd_grad/BiasAddGrad
�
@lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependencyIdentity-lenet1/gradients/lenet1/h3/Relu_grad/ReluGrad9^lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@lenet1/gradients/lenet1/h3/Relu_grad/ReluGrad*/
_output_shapes
:���������
�
Blenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependency_1Identity3lenet1/gradients/lenet1/h3/BiasAdd_grad/BiasAddGrad9^lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@lenet1/gradients/lenet1/h3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
1lenet1/gradients/lenet1/h3/convolution_grad/ShapeShapelenet1/h2/AvgPool*
T0*
_output_shapes
:*
out_type0
�
?lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropInputConv2DBackpropInput1lenet1/gradients/lenet1/h3/convolution_grad/Shapelenet1/h3/kernel/read@lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC
�
3lenet1/gradients/lenet1/h3/convolution_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
�
@lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterlenet1/h2/AvgPool3lenet1/gradients/lenet1/h3/convolution_grad/Shape_1@lenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
:*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC
�
<lenet1/gradients/lenet1/h3/convolution_grad/tuple/group_depsNoOp@^lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropInputA^lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropFilter
�
Dlenet1/gradients/lenet1/h3/convolution_grad/tuple/control_dependencyIdentity?lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropInput=^lenet1/gradients/lenet1/h3/convolution_grad/tuple/group_deps*
T0*R
_classH
FDloc:@lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
Flenet1/gradients/lenet1/h3/convolution_grad/tuple/control_dependency_1Identity@lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropFilter=^lenet1/gradients/lenet1/h3/convolution_grad/tuple/group_deps*
T0*S
_classI
GEloc:@lenet1/gradients/lenet1/h3/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:
{
-lenet1/gradients/lenet1/h2/AvgPool_grad/ShapeShapelenet1/h1/Relu*
T0*
_output_shapes
:*
out_type0
�
3lenet1/gradients/lenet1/h2/AvgPool_grad/AvgPoolGradAvgPoolGrad-lenet1/gradients/lenet1/h2/AvgPool_grad/ShapeDlenet1/gradients/lenet1/h3/convolution_grad/tuple/control_dependency*/
_output_shapes
:���������*
paddingVALID*
ksize
*
T0*
strides
*
data_formatNHWC
�
-lenet1/gradients/lenet1/h1/Relu_grad/ReluGradReluGrad3lenet1/gradients/lenet1/h2/AvgPool_grad/AvgPoolGradlenet1/h1/Relu*
T0*/
_output_shapes
:���������
�
3lenet1/gradients/lenet1/h1/BiasAdd_grad/BiasAddGradBiasAddGrad-lenet1/gradients/lenet1/h1/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
�
8lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/group_depsNoOp.^lenet1/gradients/lenet1/h1/Relu_grad/ReluGrad4^lenet1/gradients/lenet1/h1/BiasAdd_grad/BiasAddGrad
�
@lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependencyIdentity-lenet1/gradients/lenet1/h1/Relu_grad/ReluGrad9^lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@lenet1/gradients/lenet1/h1/Relu_grad/ReluGrad*/
_output_shapes
:���������
�
Blenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependency_1Identity3lenet1/gradients/lenet1/h1/BiasAdd_grad/BiasAddGrad9^lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@lenet1/gradients/lenet1/h1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

1lenet1/gradients/lenet1/h1/convolution_grad/ShapeShapelenet1/Reshape*
T0*
_output_shapes
:*
out_type0
�
?lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput1lenet1/gradients/lenet1/h1/convolution_grad/Shapelenet1/h1/kernel/read@lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependency*
strides
*J
_output_shapes8
6:4������������������������������������*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC
�
3lenet1/gradients/lenet1/h1/convolution_grad/Shape_1Const*%
valueB"            *
_output_shapes
:*
dtype0
�
@lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterlenet1/Reshape3lenet1/gradients/lenet1/h1/convolution_grad/Shape_1@lenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependency*
strides
*&
_output_shapes
:*
paddingVALID*
T0*
use_cudnn_on_gpu(*
data_formatNHWC
�
<lenet1/gradients/lenet1/h1/convolution_grad/tuple/group_depsNoOp@^lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropInputA^lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropFilter
�
Dlenet1/gradients/lenet1/h1/convolution_grad/tuple/control_dependencyIdentity?lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropInput=^lenet1/gradients/lenet1/h1/convolution_grad/tuple/group_deps*
T0*R
_classH
FDloc:@lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:���������
�
Flenet1/gradients/lenet1/h1/convolution_grad/tuple/control_dependency_1Identity@lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropFilter=^lenet1/gradients/lenet1/h1/convolution_grad/tuple/group_deps*
T0*S
_classI
GEloc:@lenet1/gradients/lenet1/h1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
 lenet1/beta1_power/initial_valueConst*#
_class
loc:@lenet1/h1/kernel*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
lenet1/beta1_power
VariableV2*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 
�
lenet1/beta1_power/AssignAssignlenet1/beta1_power lenet1/beta1_power/initial_value*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking(
}
lenet1/beta1_power/readIdentitylenet1/beta1_power*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
 lenet1/beta2_power/initial_valueConst*#
_class
loc:@lenet1/h1/kernel*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
lenet1/beta2_power
VariableV2*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 
�
lenet1/beta2_power/AssignAssignlenet1/beta2_power lenet1/beta2_power/initial_value*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking(
}
lenet1/beta2_power/readIdentitylenet1/beta2_power*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
.lenet1/lenet1/h1/kernel/Adam/Initializer/ConstConst*#
_class
loc:@lenet1/h1/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
�
lenet1/lenet1/h1/kernel/Adam
VariableV2*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
#lenet1/lenet1/h1/kernel/Adam/AssignAssignlenet1/lenet1/h1/kernel/Adam.lenet1/lenet1/h1/kernel/Adam/Initializer/Const*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
!lenet1/lenet1/h1/kernel/Adam/readIdentitylenet1/lenet1/h1/kernel/Adam*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
0lenet1/lenet1/h1/kernel/Adam_1/Initializer/ConstConst*#
_class
loc:@lenet1/h1/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
�
lenet1/lenet1/h1/kernel/Adam_1
VariableV2*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
%lenet1/lenet1/h1/kernel/Adam_1/AssignAssignlenet1/lenet1/h1/kernel/Adam_10lenet1/lenet1/h1/kernel/Adam_1/Initializer/Const*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
#lenet1/lenet1/h1/kernel/Adam_1/readIdentitylenet1/lenet1/h1/kernel/Adam_1*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:
�
,lenet1/lenet1/h1/bias/Adam/Initializer/ConstConst*!
_class
loc:@lenet1/h1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/lenet1/h1/bias/Adam
VariableV2*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
!lenet1/lenet1/h1/bias/Adam/AssignAssignlenet1/lenet1/h1/bias/Adam,lenet1/lenet1/h1/bias/Adam/Initializer/Const*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
lenet1/lenet1/h1/bias/Adam/readIdentitylenet1/lenet1/h1/bias/Adam*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:
�
.lenet1/lenet1/h1/bias/Adam_1/Initializer/ConstConst*!
_class
loc:@lenet1/h1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/lenet1/h1/bias/Adam_1
VariableV2*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
#lenet1/lenet1/h1/bias/Adam_1/AssignAssignlenet1/lenet1/h1/bias/Adam_1.lenet1/lenet1/h1/bias/Adam_1/Initializer/Const*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
!lenet1/lenet1/h1/bias/Adam_1/readIdentitylenet1/lenet1/h1/bias/Adam_1*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:
�
.lenet1/lenet1/h3/kernel/Adam/Initializer/ConstConst*#
_class
loc:@lenet1/h3/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
�
lenet1/lenet1/h3/kernel/Adam
VariableV2*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
#lenet1/lenet1/h3/kernel/Adam/AssignAssignlenet1/lenet1/h3/kernel/Adam.lenet1/lenet1/h3/kernel/Adam/Initializer/Const*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
!lenet1/lenet1/h3/kernel/Adam/readIdentitylenet1/lenet1/h3/kernel/Adam*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
0lenet1/lenet1/h3/kernel/Adam_1/Initializer/ConstConst*#
_class
loc:@lenet1/h3/kernel*%
valueB*    *
dtype0*&
_output_shapes
:
�
lenet1/lenet1/h3/kernel/Adam_1
VariableV2*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
%lenet1/lenet1/h3/kernel/Adam_1/AssignAssignlenet1/lenet1/h3/kernel/Adam_10lenet1/lenet1/h3/kernel/Adam_1/Initializer/Const*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
#lenet1/lenet1/h3/kernel/Adam_1/readIdentitylenet1/lenet1/h3/kernel/Adam_1*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:
�
,lenet1/lenet1/h3/bias/Adam/Initializer/ConstConst*!
_class
loc:@lenet1/h3/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/lenet1/h3/bias/Adam
VariableV2*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
!lenet1/lenet1/h3/bias/Adam/AssignAssignlenet1/lenet1/h3/bias/Adam,lenet1/lenet1/h3/bias/Adam/Initializer/Const*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
lenet1/lenet1/h3/bias/Adam/readIdentitylenet1/lenet1/h3/bias/Adam*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:
�
.lenet1/lenet1/h3/bias/Adam_1/Initializer/ConstConst*!
_class
loc:@lenet1/h3/bias*
valueB*    *
dtype0*
_output_shapes
:
�
lenet1/lenet1/h3/bias/Adam_1
VariableV2*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
�
#lenet1/lenet1/h3/bias/Adam_1/AssignAssignlenet1/lenet1/h3/bias/Adam_1.lenet1/lenet1/h3/bias/Adam_1/Initializer/Const*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
!lenet1/lenet1/h3/bias/Adam_1/readIdentitylenet1/lenet1/h3/bias/Adam_1*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:
�
3lenet1/lenet1/outputs/kernel/Adam/Initializer/ConstConst*(
_class
loc:@lenet1/outputs/kernel*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
!lenet1/lenet1/outputs/kernel/Adam
VariableV2*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
shape:	�
*
	container *
dtype0*
shared_name 
�
(lenet1/lenet1/outputs/kernel/Adam/AssignAssign!lenet1/lenet1/outputs/kernel/Adam3lenet1/lenet1/outputs/kernel/Adam/Initializer/Const*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
&lenet1/lenet1/outputs/kernel/Adam/readIdentity!lenet1/lenet1/outputs/kernel/Adam*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
5lenet1/lenet1/outputs/kernel/Adam_1/Initializer/ConstConst*(
_class
loc:@lenet1/outputs/kernel*
valueB	�
*    *
dtype0*
_output_shapes
:	�

�
#lenet1/lenet1/outputs/kernel/Adam_1
VariableV2*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
shape:	�
*
	container *
dtype0*
shared_name 
�
*lenet1/lenet1/outputs/kernel/Adam_1/AssignAssign#lenet1/lenet1/outputs/kernel/Adam_15lenet1/lenet1/outputs/kernel/Adam_1/Initializer/Const*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
(lenet1/lenet1/outputs/kernel/Adam_1/readIdentity#lenet1/lenet1/outputs/kernel/Adam_1*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�

�
1lenet1/lenet1/outputs/bias/Adam/Initializer/ConstConst*&
_class
loc:@lenet1/outputs/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
lenet1/lenet1/outputs/bias/Adam
VariableV2*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
shape:
*
	container *
dtype0*
shared_name 
�
&lenet1/lenet1/outputs/bias/Adam/AssignAssignlenet1/lenet1/outputs/bias/Adam1lenet1/lenet1/outputs/bias/Adam/Initializer/Const*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
$lenet1/lenet1/outputs/bias/Adam/readIdentitylenet1/lenet1/outputs/bias/Adam*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:

�
3lenet1/lenet1/outputs/bias/Adam_1/Initializer/ConstConst*&
_class
loc:@lenet1/outputs/bias*
valueB
*    *
dtype0*
_output_shapes
:

�
!lenet1/lenet1/outputs/bias/Adam_1
VariableV2*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
shape:
*
	container *
dtype0*
shared_name 
�
(lenet1/lenet1/outputs/bias/Adam_1/AssignAssign!lenet1/lenet1/outputs/bias/Adam_13lenet1/lenet1/outputs/bias/Adam_1/Initializer/Const*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
&lenet1/lenet1/outputs/bias/Adam_1/readIdentity!lenet1/lenet1/outputs/bias/Adam_1*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:

^
lenet1/Adam/learning_rateConst*
valueB
 *o�:*
_output_shapes
: *
dtype0
V
lenet1/Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
V
lenet1/Adam/beta2Const*
valueB
 *w�?*
_output_shapes
: *
dtype0
X
lenet1/Adam/epsilonConst*
valueB
 *w�+2*
_output_shapes
: *
dtype0
�
-lenet1/Adam/update_lenet1/h1/kernel/ApplyAdam	ApplyAdamlenet1/h1/kernellenet1/lenet1/h1/kernel/Adamlenet1/lenet1/h1/kernel/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonFlenet1/gradients/lenet1/h1/convolution_grad/tuple/control_dependency_1*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
use_locking( 
�
+lenet1/Adam/update_lenet1/h1/bias/ApplyAdam	ApplyAdamlenet1/h1/biaslenet1/lenet1/h1/bias/Adamlenet1/lenet1/h1/bias/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonBlenet1/gradients/lenet1/h1/BiasAdd_grad/tuple/control_dependency_1*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
use_locking( 
�
-lenet1/Adam/update_lenet1/h3/kernel/ApplyAdam	ApplyAdamlenet1/h3/kernellenet1/lenet1/h3/kernel/Adamlenet1/lenet1/h3/kernel/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonFlenet1/gradients/lenet1/h3/convolution_grad/tuple/control_dependency_1*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
use_locking( 
�
+lenet1/Adam/update_lenet1/h3/bias/ApplyAdam	ApplyAdamlenet1/h3/biaslenet1/lenet1/h3/bias/Adamlenet1/lenet1/h3/bias/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonBlenet1/gradients/lenet1/h3/BiasAdd_grad/tuple/control_dependency_1*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
use_locking( 
�
2lenet1/Adam/update_lenet1/outputs/kernel/ApplyAdam	ApplyAdamlenet1/outputs/kernel!lenet1/lenet1/outputs/kernel/Adam#lenet1/lenet1/outputs/kernel/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonFlenet1/gradients/lenet1/outputs/MatMul_grad/tuple/control_dependency_1*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
use_locking( 
�
0lenet1/Adam/update_lenet1/outputs/bias/ApplyAdam	ApplyAdamlenet1/outputs/biaslenet1/lenet1/outputs/bias/Adam!lenet1/lenet1/outputs/bias/Adam_1lenet1/beta1_power/readlenet1/beta2_power/readlenet1/Adam/learning_ratelenet1/Adam/beta1lenet1/Adam/beta2lenet1/Adam/epsilonGlenet1/gradients/lenet1/outputs/BiasAdd_grad/tuple/control_dependency_1*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
use_locking( 
�
lenet1/Adam/mulMullenet1/beta1_power/readlenet1/Adam/beta1.^lenet1/Adam/update_lenet1/h1/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h1/bias/ApplyAdam.^lenet1/Adam/update_lenet1/h3/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h3/bias/ApplyAdam3^lenet1/Adam/update_lenet1/outputs/kernel/ApplyAdam1^lenet1/Adam/update_lenet1/outputs/bias/ApplyAdam*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
lenet1/Adam/AssignAssignlenet1/beta1_powerlenet1/Adam/mul*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking( 
�
lenet1/Adam/mul_1Mullenet1/beta2_power/readlenet1/Adam/beta2.^lenet1/Adam/update_lenet1/h1/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h1/bias/ApplyAdam.^lenet1/Adam/update_lenet1/h3/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h3/bias/ApplyAdam3^lenet1/Adam/update_lenet1/outputs/kernel/ApplyAdam1^lenet1/Adam/update_lenet1/outputs/bias/ApplyAdam*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: 
�
lenet1/Adam/Assign_1Assignlenet1/beta2_powerlenet1/Adam/mul_1*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking( 
�
lenet1/Adam/updateNoOp.^lenet1/Adam/update_lenet1/h1/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h1/bias/ApplyAdam.^lenet1/Adam/update_lenet1/h3/kernel/ApplyAdam,^lenet1/Adam/update_lenet1/h3/bias/ApplyAdam3^lenet1/Adam/update_lenet1/outputs/kernel/ApplyAdam1^lenet1/Adam/update_lenet1/outputs/bias/ApplyAdam^lenet1/Adam/Assign^lenet1/Adam/Assign_1
�
lenet1/Adam/valueConst^lenet1/Adam/update*%
_class
loc:@lenet1/global_step*
value	B	 R*
dtype0	*
_output_shapes
: 
�
lenet1/Adam	AssignAddlenet1/global_steplenet1/Adam/value*
T0	*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
use_locking( 
Y
lenet1/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
�
lenet1/ArgMaxArgMaxlenet1/outputs/Sigmoidlenet1/ArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:���������
[
lenet1/ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
}
lenet1/ArgMax_1ArgMaxlenet1/labelslenet1/ArgMax_1/dimension*
T0*

Tidx0*#
_output_shapes
:���������
c
lenet1/EqualEquallenet1/ArgMaxlenet1/ArgMax_1*
T0	*#
_output_shapes
:���������
`
lenet1/Cast_1Castlenet1/Equal*#
_output_shapes
:���������*

SrcT0
*

DstT0
X
lenet1/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
t
lenet1/accuracyMeanlenet1/Cast_1lenet1/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
p
lenet1/train_accuracy/tagsConst*&
valueB Blenet1/train_accuracy*
_output_shapes
: *
dtype0
t
lenet1/train_accuracyScalarSummarylenet1/train_accuracy/tagslenet1/accuracy*
T0*
_output_shapes
: 
n
lenet1/Merge/MergeSummaryMergeSummarylenet1/losslenet1/train_accuracy*
_output_shapes
: *
N
�
initNoOp^lenet1/h1/kernel/Assign^lenet1/h1/bias/Assign^lenet1/h3/kernel/Assign^lenet1/h3/bias/Assign^lenet1/outputs/kernel/Assign^lenet1/outputs/bias/Assign^lenet1/global_step/Assign^lenet1/beta1_power/Assign^lenet1/beta2_power/Assign$^lenet1/lenet1/h1/kernel/Adam/Assign&^lenet1/lenet1/h1/kernel/Adam_1/Assign"^lenet1/lenet1/h1/bias/Adam/Assign$^lenet1/lenet1/h1/bias/Adam_1/Assign$^lenet1/lenet1/h3/kernel/Adam/Assign&^lenet1/lenet1/h3/kernel/Adam_1/Assign"^lenet1/lenet1/h3/bias/Adam/Assign$^lenet1/lenet1/h3/bias/Adam_1/Assign)^lenet1/lenet1/outputs/kernel/Adam/Assign+^lenet1/lenet1/outputs/kernel/Adam_1/Assign'^lenet1/lenet1/outputs/bias/Adam/Assign)^lenet1/lenet1/outputs/bias/Adam_1/Assign
P

save/ConstConst*
valueB Bmodel*
_output_shapes
: *
dtype0
�
save/SaveV2/tensor_namesConst*�
value�B�Blenet1/beta1_powerBlenet1/beta2_powerBlenet1/global_stepBlenet1/h1/biasBlenet1/h1/kernelBlenet1/h3/biasBlenet1/h3/kernelBlenet1/lenet1/h1/bias/AdamBlenet1/lenet1/h1/bias/Adam_1Blenet1/lenet1/h1/kernel/AdamBlenet1/lenet1/h1/kernel/Adam_1Blenet1/lenet1/h3/bias/AdamBlenet1/lenet1/h3/bias/Adam_1Blenet1/lenet1/h3/kernel/AdamBlenet1/lenet1/h3/kernel/Adam_1Blenet1/lenet1/outputs/bias/AdamB!lenet1/lenet1/outputs/bias/Adam_1B!lenet1/lenet1/outputs/kernel/AdamB#lenet1/lenet1/outputs/kernel/Adam_1Blenet1/outputs/biasBlenet1/outputs/kernel*
_output_shapes
:*
dtype0
�
save/SaveV2/shape_and_slicesConst*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslenet1/beta1_powerlenet1/beta2_powerlenet1/global_steplenet1/h1/biaslenet1/h1/kernellenet1/h3/biaslenet1/h3/kernellenet1/lenet1/h1/bias/Adamlenet1/lenet1/h1/bias/Adam_1lenet1/lenet1/h1/kernel/Adamlenet1/lenet1/h1/kernel/Adam_1lenet1/lenet1/h3/bias/Adamlenet1/lenet1/h3/bias/Adam_1lenet1/lenet1/h3/kernel/Adamlenet1/lenet1/h3/kernel/Adam_1lenet1/lenet1/outputs/bias/Adam!lenet1/lenet1/outputs/bias/Adam_1!lenet1/lenet1/outputs/kernel/Adam#lenet1/lenet1/outputs/kernel/Adam_1lenet1/outputs/biaslenet1/outputs/kernel*#
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
v
save/RestoreV2/tensor_namesConst*'
valueBBlenet1/beta1_power*
_output_shapes
:*
dtype0
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignlenet1/beta1_powersave/RestoreV2*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking(
x
save/RestoreV2_1/tensor_namesConst*'
valueBBlenet1/beta2_power*
_output_shapes
:*
dtype0
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assignlenet1/beta2_powersave/RestoreV2_1*
T0*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
validate_shape(*
use_locking(
x
save/RestoreV2_2/tensor_namesConst*'
valueBBlenet1/global_step*
_output_shapes
:*
dtype0
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2	*
_output_shapes
:
�
save/Assign_2Assignlenet1/global_stepsave/RestoreV2_2*
T0	*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
validate_shape(*
use_locking(
t
save/RestoreV2_3/tensor_namesConst*#
valueBBlenet1/h1/bias*
_output_shapes
:*
dtype0
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignlenet1/h1/biassave/RestoreV2_3*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
v
save/RestoreV2_4/tensor_namesConst*%
valueBBlenet1/h1/kernel*
_output_shapes
:*
dtype0
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_4Assignlenet1/h1/kernelsave/RestoreV2_4*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
t
save/RestoreV2_5/tensor_namesConst*#
valueBBlenet1/h3/bias*
_output_shapes
:*
dtype0
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_5Assignlenet1/h3/biassave/RestoreV2_5*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
v
save/RestoreV2_6/tensor_namesConst*%
valueBBlenet1/h3/kernel*
_output_shapes
:*
dtype0
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_6Assignlenet1/h3/kernelsave/RestoreV2_6*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_7/tensor_namesConst*/
value&B$Blenet1/lenet1/h1/bias/Adam*
_output_shapes
:*
dtype0
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_7Assignlenet1/lenet1/h1/bias/Adamsave/RestoreV2_7*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_8/tensor_namesConst*1
value(B&Blenet1/lenet1/h1/bias/Adam_1*
_output_shapes
:*
dtype0
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assignlenet1/lenet1/h1/bias/Adam_1save/RestoreV2_8*
T0*!
_class
loc:@lenet1/h1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_9/tensor_namesConst*1
value(B&Blenet1/lenet1/h1/kernel/Adam*
_output_shapes
:*
dtype0
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assignlenet1/lenet1/h1/kernel/Adamsave/RestoreV2_9*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_10/tensor_namesConst*3
value*B(Blenet1/lenet1/h1/kernel/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_10Assignlenet1/lenet1/h1/kernel/Adam_1save/RestoreV2_10*
T0*#
_class
loc:@lenet1/h1/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_11/tensor_namesConst*/
value&B$Blenet1/lenet1/h3/bias/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_11Assignlenet1/lenet1/h3/bias/Adamsave/RestoreV2_11*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_12/tensor_namesConst*1
value(B&Blenet1/lenet1/h3/bias/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_12Assignlenet1/lenet1/h3/bias/Adam_1save/RestoreV2_12*
T0*!
_class
loc:@lenet1/h3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_13/tensor_namesConst*1
value(B&Blenet1/lenet1/h3/kernel/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_13Assignlenet1/lenet1/h3/kernel/Adamsave/RestoreV2_13*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_14/tensor_namesConst*3
value*B(Blenet1/lenet1/h3/kernel/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assignlenet1/lenet1/h3/kernel/Adam_1save/RestoreV2_14*
T0*#
_class
loc:@lenet1/h3/kernel*&
_output_shapes
:*
validate_shape(*
use_locking(
�
save/RestoreV2_15/tensor_namesConst*4
value+B)Blenet1/lenet1/outputs/bias/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_15Assignlenet1/lenet1/outputs/bias/Adamsave/RestoreV2_15*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
save/RestoreV2_16/tensor_namesConst*6
value-B+B!lenet1/lenet1/outputs/bias/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_16Assign!lenet1/lenet1/outputs/bias/Adam_1save/RestoreV2_16*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
�
save/RestoreV2_17/tensor_namesConst*6
value-B+B!lenet1/lenet1/outputs/kernel/Adam*
_output_shapes
:*
dtype0
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_17Assign!lenet1/lenet1/outputs/kernel/Adamsave/RestoreV2_17*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
save/RestoreV2_18/tensor_namesConst*8
value/B-B#lenet1/lenet1/outputs/kernel/Adam_1*
_output_shapes
:*
dtype0
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assign#lenet1/lenet1/outputs/kernel/Adam_1save/RestoreV2_18*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
z
save/RestoreV2_19/tensor_namesConst*(
valueBBlenet1/outputs/bias*
_output_shapes
:*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_19Assignlenet1/outputs/biassave/RestoreV2_19*
T0*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
:
*
validate_shape(*
use_locking(
|
save/RestoreV2_20/tensor_namesConst**
value!BBlenet1/outputs/kernel*
_output_shapes
:*
dtype0
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
_output_shapes
:*
dtype0
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assignlenet1/outputs/kernelsave/RestoreV2_20*
T0*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
:	�
*
validate_shape(*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20
�
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedlenet1/h1/kernel*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedlenet1/h1/bias*!
_class
loc:@lenet1/h1/bias*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedlenet1/h3/kernel*#
_class
loc:@lenet1/h3/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedlenet1/h3/bias*!
_class
loc:@lenet1/h3/bias*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedlenet1/outputs/kernel*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedlenet1/outputs/bias*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedlenet1/global_step*%
_class
loc:@lenet1/global_step*
_output_shapes
: *
dtype0	
�
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedlenet1/beta1_power*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializedlenet1/beta2_power*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedlenet1/lenet1/h1/kernel/Adam*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedlenet1/lenet1/h1/kernel/Adam_1*#
_class
loc:@lenet1/h1/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializedlenet1/lenet1/h1/bias/Adam*!
_class
loc:@lenet1/h1/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializedlenet1/lenet1/h1/bias/Adam_1*!
_class
loc:@lenet1/h1/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializedlenet1/lenet1/h3/kernel/Adam*#
_class
loc:@lenet1/h3/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializedlenet1/lenet1/h3/kernel/Adam_1*#
_class
loc:@lenet1/h3/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedlenet1/lenet1/h3/bias/Adam*!
_class
loc:@lenet1/h3/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedlenet1/lenet1/h3/bias/Adam_1*!
_class
loc:@lenet1/h3/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized!lenet1/lenet1/outputs/kernel/Adam*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized#lenet1/lenet1/outputs/kernel/Adam_1*(
_class
loc:@lenet1/outputs/kernel*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitializedlenet1/lenet1/outputs/bias/Adam*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
: *
dtype0
�
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized!lenet1/lenet1/outputs/bias/Adam_1*&
_class
loc:@lenet1/outputs/bias*
_output_shapes
: *
dtype0
�

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_20*
T0
*

axis *
_output_shapes
:*
N
y
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack*
_output_shapes
:
�
$report_uninitialized_variables/ConstConst*�
value�B�Blenet1/h1/kernelBlenet1/h1/biasBlenet1/h3/kernelBlenet1/h3/biasBlenet1/outputs/kernelBlenet1/outputs/biasBlenet1/global_stepBlenet1/beta1_powerBlenet1/beta2_powerBlenet1/lenet1/h1/kernel/AdamBlenet1/lenet1/h1/kernel/Adam_1Blenet1/lenet1/h1/bias/AdamBlenet1/lenet1/h1/bias/Adam_1Blenet1/lenet1/h3/kernel/AdamBlenet1/lenet1/h3/kernel/Adam_1Blenet1/lenet1/h3/bias/AdamBlenet1/lenet1/h3/bias/Adam_1B!lenet1/lenet1/outputs/kernel/AdamB#lenet1/lenet1/outputs/kernel/Adam_1Blenet1/lenet1/outputs/bias/AdamB!lenet1/lenet1/outputs/bias/Adam_1*
_output_shapes
:*
dtype0
{
1report_uninitialized_variables/boolean_mask/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
�
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2*
ellipsis_mask *

begin_mask*
_output_shapes
:*
end_mask *
new_axis_mask *
T0*
Index0*
shrink_axis_mask 
�
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst*
valueB: *
_output_shapes
:*
dtype0
�
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
}
3report_uninitialized_variables/boolean_mask/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst*
valueB:*
_output_shapes
:*
dtype0
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
_output_shapes
:*
dtype0
�
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
�
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask *
_output_shapes
: *
end_mask*
new_axis_mask *
T0*
Index0*
shrink_axis_mask 
�
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod*
T0*

axis *
_output_shapes
:*
N
y
7report_uninitialized_variables/boolean_mask/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis*
T0*

Tidx0*
_output_shapes
:*
N
�
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat*
T0*
_output_shapes
:*
Tshape0
�
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst*
valueB:
���������*
_output_shapes
:*
dtype0
�
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape*
T0
*
_output_shapes
:*
Tshape0
�
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1*'
_output_shapes
:���������
�
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where*
T0	*
squeeze_dims
*#
_output_shapes
:���������
�
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze*#
_output_shapes
:���������*
Tparams0*
validate_indices(*
Tindices0	

init_1NoOp

init_all_tablesNoOp
-

group_depsNoOp^init_1^init_all_tables""D
ready_op8
6
4report_uninitialized_variables/boolean_mask/Gather:0"[
neuronsP
N
lenet1/h1/Relu:0
lenet1/h2/AvgPool:0
lenet1/h3/Relu:0
lenet1/h4/AvgPool:0"�
trainable_variables��
F
lenet1/h1/kernel:0lenet1/h1/kernel/Assignlenet1/h1/kernel/read:0
@
lenet1/h1/bias:0lenet1/h1/bias/Assignlenet1/h1/bias/read:0
F
lenet1/h3/kernel:0lenet1/h3/kernel/Assignlenet1/h3/kernel/read:0
@
lenet1/h3/bias:0lenet1/h3/bias/Assignlenet1/h3/bias/read:0
U
lenet1/outputs/kernel:0lenet1/outputs/kernel/Assignlenet1/outputs/kernel/read:0
O
lenet1/outputs/bias:0lenet1/outputs/bias/Assignlenet1/outputs/bias/read:0"7
	summaries*
(
lenet1/loss:0
lenet1/train_accuracy:0"
train_op

lenet1/Adam"
local_init_op


group_deps"�
	variables��
F
lenet1/h1/kernel:0lenet1/h1/kernel/Assignlenet1/h1/kernel/read:0
@
lenet1/h1/bias:0lenet1/h1/bias/Assignlenet1/h1/bias/read:0
F
lenet1/h3/kernel:0lenet1/h3/kernel/Assignlenet1/h3/kernel/read:0
@
lenet1/h3/bias:0lenet1/h3/bias/Assignlenet1/h3/bias/read:0
U
lenet1/outputs/kernel:0lenet1/outputs/kernel/Assignlenet1/outputs/kernel/read:0
O
lenet1/outputs/bias:0lenet1/outputs/bias/Assignlenet1/outputs/bias/read:0
L
lenet1/global_step:0lenet1/global_step/Assignlenet1/global_step/read:0
L
lenet1/beta1_power:0lenet1/beta1_power/Assignlenet1/beta1_power/read:0
L
lenet1/beta2_power:0lenet1/beta2_power/Assignlenet1/beta2_power/read:0
j
lenet1/lenet1/h1/kernel/Adam:0#lenet1/lenet1/h1/kernel/Adam/Assign#lenet1/lenet1/h1/kernel/Adam/read:0
p
 lenet1/lenet1/h1/kernel/Adam_1:0%lenet1/lenet1/h1/kernel/Adam_1/Assign%lenet1/lenet1/h1/kernel/Adam_1/read:0
d
lenet1/lenet1/h1/bias/Adam:0!lenet1/lenet1/h1/bias/Adam/Assign!lenet1/lenet1/h1/bias/Adam/read:0
j
lenet1/lenet1/h1/bias/Adam_1:0#lenet1/lenet1/h1/bias/Adam_1/Assign#lenet1/lenet1/h1/bias/Adam_1/read:0
j
lenet1/lenet1/h3/kernel/Adam:0#lenet1/lenet1/h3/kernel/Adam/Assign#lenet1/lenet1/h3/kernel/Adam/read:0
p
 lenet1/lenet1/h3/kernel/Adam_1:0%lenet1/lenet1/h3/kernel/Adam_1/Assign%lenet1/lenet1/h3/kernel/Adam_1/read:0
d
lenet1/lenet1/h3/bias/Adam:0!lenet1/lenet1/h3/bias/Adam/Assign!lenet1/lenet1/h3/bias/Adam/read:0
j
lenet1/lenet1/h3/bias/Adam_1:0#lenet1/lenet1/h3/bias/Adam_1/Assign#lenet1/lenet1/h3/bias/Adam_1/read:0
y
#lenet1/lenet1/outputs/kernel/Adam:0(lenet1/lenet1/outputs/kernel/Adam/Assign(lenet1/lenet1/outputs/kernel/Adam/read:0

%lenet1/lenet1/outputs/kernel/Adam_1:0*lenet1/lenet1/outputs/kernel/Adam_1/Assign*lenet1/lenet1/outputs/kernel/Adam_1/read:0
s
!lenet1/lenet1/outputs/bias/Adam:0&lenet1/lenet1/outputs/bias/Adam/Assign&lenet1/lenet1/outputs/bias/Adam/read:0
y
#lenet1/lenet1/outputs/bias/Adam_1:0(lenet1/lenet1/outputs/bias/Adam_1/Assign(lenet1/lenet1/outputs/bias/Adam_1/read:0"'
global_step

lenet1/global_step:0#��=       `I��	�J����A*2

lenet1/loss��@

lenet1/train_accuracy���=����@       (��	��騖��A�*2

lenet1/loss�2�?

lenet1/train_accuracy)\O?�Fd�@       (��	�Md����A�*2

lenet1/loss ��?

lenet1/train_accuracy��h?�S��@       (��	��௖��A�	*2

lenet1/loss� �?

lenet1/train_accuracy�k?��d@       (��	�A]����A�*2

lenet1/lossw��?

lenet1/train_accuracy33s?�2g@       (��	��޶���A�*2

lenet1/lossC��?

lenet1/train_accuracyףp?�!�@       (��	OXZ����A�*2

lenet1/lossщ�?

lenet1/train_accuracyףp?^4M=@       (��	t�ս���A�*2

lenet1/loss���?

lenet1/train_accuracy�k?��5�@       (��	b�Q����A�*2

lenet1/loss��?

lenet1/train_accuracy33s?�d�e@       (��	���Ė��A�*2

lenet1/loss��?

lenet1/train_accuracy�p}?�]�@       (��	CȖ��A�*2

lenet1/lossA־?

lenet1/train_accuracy��u?�+�@       (��	G��˖��A�"*2

lenet1/loss}j�?

lenet1/train_accuracy33s?�߿7@       (��	�wUϖ��A�%*2

lenet1/lossT��?

lenet1/train_accuracy��u?���@       (��	�E�Җ��A�(*2

lenet1/loss���?

lenet1/train_accuracy33s?"';�@       (��	]�W֖��A�+*2

lenet1/lossM	�?

lenet1/train_accuracy�Qx?��d@       (��	��ٖ��A�.*2

lenet1/lossa��?

lenet1/train_accuracyH�z?�
@       (��	L�Dݖ��A�2*2

lenet1/loss�ӽ?

lenet1/train_accuracy�Qx?z��@       (��	Dj�����A�5*2

lenet1/loss.�?

lenet1/train_accuracyH�z?��kv@       (��	��7䖀�A�8*2

lenet1/loss��?

lenet1/train_accuracyH�z?|�ه@       (��	�?�疀�A�;*2

lenet1/loss���?

lenet1/train_accuracy33s?�\��@       (��	>z%떀�A�>*2

lenet1/loss{��?

lenet1/train_accuracy�p}?��@       (��	�U�A�A*2

lenet1/loss�*�?

lenet1/train_accuracy�p}?�vB@       (��	�����A�D*2

lenet1/lossV�?

lenet1/train_accuracy��u?��H�@       (��	�Y����A�G*2

lenet1/loss�	�?

lenet1/train_accuracy��u?�@�@       (��	S�����A�K*2

lenet1/loss�ʾ?

lenet1/train_accuracyH�z?ChН@       (��	�ʋ����A�N*2

lenet1/loss���?

lenet1/train_accuracy33s?�j"@