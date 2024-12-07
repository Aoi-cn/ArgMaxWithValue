
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgMaxWithValueTilingData){
  TILING_DATA_FIELD_DEF(uint32_t, lx);
  TILING_DATA_FIELD_DEF(uint32_t, ly);
  TILING_DATA_FIELD_DEF(uint32_t, lz);
  TILING_DATA_FIELD_DEF(uint32_t, dim);
  TILING_DATA_FIELD_DEF(uint32_t, dimNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  // TILING_DATA_FIELD_DEF(uint32_t, blockNum);  //数据占有的block个数
  // TILING_DATA_FIELD_DEF(uint32_t, ubSize);  //UB大小

  //保留大小核优化
  // TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);  //小核的数据总量  16x10=160
  // TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);  //大核处理的数据总量  16x11=176
  // TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum); //大核搬运次数：11/8=1->2
  // TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum); //小核搬运次数：10/8=1->2
  
  // TILING_DATA_FIELD_DEF(uint32_t, samllTailDataNum);  //最后一次搬运数据数量：2x16=32
  // TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);  //最后一次搬运数量：3x16=48
  // TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);  //大核个数
}
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgMaxWithValue, ArgMaxWithValueTilingData)
}
