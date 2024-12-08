
#include "arg_max_with_value_tiling.h"
#include "register/op_def_registry.h"

const uint32_t BLOCK_SIZE=32;
const uint32_t BUFFER_NUM=2;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ArgMaxWithValueTilingData tiling;
  uint64_t ubSize;
  auto dim=*context->GetAttrs()->GetInt(0);
  
  auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubSize);
  //auto coreNum=ascendPlatform.GetCoreNum();   保留优化
  auto coreNum=1;
  uint32_t typeLength=0;
  ge::TypeUtils::GetDataTypeLength(constext->GetInputDesc(0)->GetDataType(),typeLength);
  uint32_t lx=1;
  uint32_t ly=1;
  uint32_t lz=1;
  tiling.set_lx(1);
  tiling.set_ly(1);
  tiling.set_lz(1);
  tiling.set_ubSize(ubSize);
  tiling.set_dim(dim);
  uint32_t reduceNum=1;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++){
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
    if(i==0){
        lx=x1_shape->GetStorageShape().GetDim(i);
        tiling.set_lx(x1_shape->GetStorageShape().GetDim(i));
    }
    else if(i==1){
        ly=x1_shape->GetStorageShape().GetDim(i);
        tiling.set_ly(x1_shape->GetStorageShape().GetDim(i));
    }
    else if(i==2){
        lz=x1_shape->GetStorageShape().GetDim(i);
        tiling.set_lz(x1_shape->GetStorageShape().GetDim(i));
    }
    if(i==dim){
        tiling.set_dimNum(x1_shape->GetStorageShape().GetDim(i));
    }
    if(i<x1_shape->GetStorageShape().GetDimNum()-1){
        reduceNum*=x1_shape->GetStorageShape().GetDim(i);
    }
  }
  uint32_t ubDataNumber=(typeLength==1)?4:3;
  //一次tile的block数量
  uint32_t tileBlockNum=(ubSize/BLOCK_SIZE/BUFFER_NUM)/ubDataNumber;
  //一次tile的数据个数
  uint32_t tileDataNum=tileBlockNum*BLOCK_SIZE/typeLength;
  //tile次数
  uint32_t tileNum=reduceNum/tileDataNum;
  //tail长度
  uint32_t tailLength=reduceNum%tileDataNum;
  tileNum=(tailLength==0)?tileNum:tileNum+1;

  uint32_t stride=1;
  if(dim==1){
    stride=lx;
  }
  else if(dim==2){
    stride=lx*ly;
  }
  tiling.set_tileDataNum(tileDataNum);
  tiling.set_tileNum(tileNum);
  tiling.set_tailLength(tailLength);
  tiling.set_stride(stride);

//   uint32_t inputLength=data_sz*typeLength;  //数据总B数
//   uint32_t block_num=((inputLength+BLOCK_SIZE-1)/BLOCK_SIZE)*BLOCK_SIZE;  //数据总共有多少个block个数
//   tiling.set_blockNum(block_num);

  context->SetBlockDim(8);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ArgMaxWithValue : public OpDef {
public:
    explicit ArgMaxWithValue(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indice")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dimension").Int();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(ArgMaxWithValue);
}
