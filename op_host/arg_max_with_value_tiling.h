
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgMaxWithValueTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, dimNum);
  TILING_DATA_FIELD_DEF(int, dimension);
  TILING_DATA_FIELD_DEF(bool, keep_dims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgMaxWithValue, ArgMaxWithValueTilingData)
}
