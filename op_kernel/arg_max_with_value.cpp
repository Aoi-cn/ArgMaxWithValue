#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_INDICE, typename TYPE_VALUES> class KernelArgMaxWithValue {
    using T = TYPE_X;
public:
    __aicore__ inline KernelArgMaxWithValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, int dimension, bool keep_dims, tiling_data.totalLength, tiling_data.tileNum) 
    {
        uint64_t ubSize;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        this->dimension = dimension;
        this->keep_dims = keep_dims;
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();

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


        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        indiceGm.SetGlobalBuffer((__gm__ TYPE_INDICE*)indice + globalBufferIndex, this->coreDataNum);
        valuesGm.SetGlobalBuffer((__gm__ TYPE_VALUES*)values + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueIndice, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_INDICE));
        pipe.InitBuffer(outQueueValues, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_VALUES));
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {   
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->rpParams);
        inQueueX.EnQue<TYPE_X>(xLocal);

    }
    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
      AscendC::LocalTensor<TYPE_INDICE> yLocal = outQueueIndice.DeQue<TYPE_INDICE>();
      AscendC::LocalTensor<TYPE_VALUES> zLocal = outQueueValues.AllocTensor<TYPE_VALUES>();
      if constexpr (std::is_same_v<T, int8_t>) {
        auto p1 = tmp1.Get<half>();
        auto p2 = tmp2.Get<half>();
        AscendC::Cast(p1, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(p2, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Add(p2, p1, p2, this->processDataNum);
        AscendC::Cast(p1.ReinterpretCast<int16_t>(), p2, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        AscendC::ShiftLeft(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), int16_t(8), this->processDataNum); 
        AscendC::ShiftRight(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), int16_t(8), this->processDataNum);
        AscendC::Cast(p2, p1.ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(zLocal, p2, AscendC::RoundMode::CAST_NONE, this->processDataNum);
      }
      else {
        AscendC::Add(zLocal, xLocal, yLocal, this->processDataNum);
      }
      outQueueValues.EnQue<TYPE_VALUES>(zLocal);
      inQueueX.FreeTensor(xLocal);
      outQueueIndice.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_VALUES> zLocal = outQueueValues.DeQue<TYPE_VALUES>();  
      AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
      outQueueValues.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueIndice, outQueueValues;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_INDICE> indiceGm;
    AscendC::GlobalTensor<TYPE_VALUES> valuesGm;
    int dimension;
    bool keep_dims;
    struct AscendC::DataCopyParams rpParams;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};


extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelArgMaxWithValue op;
    op.Init(x, indice, values, tiling_data.totalLength, tiling_data.tileNum);  
    op.Process();
}