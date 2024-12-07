#include "kernel_operator.h"

template<typename TYPE_X,typename TYPE_Y,typename TYPE_Z> class KernelSArg {
public:
    __aicore__ inline KernelArg() {}
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR indice,GM_ADDR values,uint32_t lx,uint32_t ly,uint32_t lz,
                                uint32_t blockNum,uint32_t ubSize/* 开发者填充参数列表 */)
    {
        //考生补充初始化代码
        ASSERT(GetBlockNum()!=0&&"block dim can not be zero!");
        this->lx=lx;
        this->ly=ly;
        this->lz=lz;
        this->blockNum=blockNum;
        this->ubSize=ubSize;
        this->blockLength=lx*ly*lz;
        // this->tileNum=tileNum;
        // ASSERT(tileNum!=0&&"tile num can not be zero!");
        // this->tileLength=this->blockLength/tileNum/BUFFER_NUM;

        //用多核优化时才考虑idx
        // xGm.SetGlobalBuffer((__gm__ TYPE_X*)x+this->blockLength*GetBlockIdx(),this->blockLength);
        // yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y+this->blockLength*GetBlockIdx(),this->blockLength);
        
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x,this->blockLength);
        if(this->dim==0){
            this->dimNum=this->lx;
            indiceGm.SetGlobalBuffer((__gm__ TYPE_Y*)indice,this->blockLength/this->lx);
            valuesGm.SetGlobalBuffer((__gm__ TYPE_Z*)values,this->blockLength/this->lx);
        }
        else if(this->dim==1){
            this->dimNum=this->ly;
            indiceGm.SetGlobalBuffer((__gm__ TYPE_Y*)indice,this->blockLength/this->ly);
            valuesGm.SetGlobalBuffer((__gm__ TYPE_Z*)values,this->blockLength/this->ly);
        }
        else{
            this->dimNum=this->lz;
            indiceGm.SetGlobalBuffer((__gm__ TYPE_Y*)indice,this->blockLength/this->lz);
            valuesGm.SetGlobalBuffer((__gm__ TYPE_Z*)values,this->blockLength/this->lz);
        }


        pipe.InitBuffer(inQueueX,BUFFER_NUM,this->tileLength*sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY,BUFFER_NUM,this->tileLength*sizeof(TYPE_Y));
        pipe.InitBuffer(tmpBuffer1,this->tileLength*sizeof(TYPE_X));
        pipe.InitBuffer(tmpBuffer2,this->tileLength*sizeof(TYPE_X));
        pipe.InitBuffer(tmpBuffer3,this->tileLength*sizeof(TYPE_X));
        pipe.InitBuffer(tmpBuffer4,this->tileLength*sizeof(TYPE_X));
        uint32_t ubDataNumber=(sizeof(TYPE_X)==1)?4:3;  //对int8的处理
        this->tileBlockNum=(this->ubSize/)
        
    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount=this->tileNum*BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<TYPE_X> xLocal=inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal,xGm[progress*this->tileLength],this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        LocalTensor<TYPE_X> xLocal=inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal=outQueueY.AllocTensor<TYPE_Y>();
        LocalTensor<TYPE_X> tmpTensor1=tmpBuffer1.Get<TYPE_X>();
        LocalTensor<TYPE_X> tmpTensor2=tmpBuffer2.Get<TYPE_X>();
        LocalTensor<TYPE_X> tmpTensor3=tmpBuffer3.Get<TYPE_X>();
        LocalTensor<TYPE_X> tmpTensor4=tmpBuffer4.Get<TYPE_X>();
        TYPE_X inputVal1=-1;
        TYPE_X inputVal2=0.5;
        Muls(tmpTensor1,xLocal,inputVal1,this->tileLength);
        Exp(tmpTensor2,tmpTensor1,this->tileLength);
        Exp(tmpTensor3,xLocal,this->tileLength);
        Sub(tmpTensor4,tmpTensor3,tmpTensor2,this->tileLength);
        Muls(yLocal,tmpTensor4,inputVal2,this->tileLength);

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<TYPE_Y>yLocal=outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress*this->tileLength],yLocal,this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueIndice,outQueueValues;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> indiceGm;
    GlobalTensor<TYPE_Z> valuesGm;
    //考生补充自定义成员变量
    TBuf<QuePosition::VECCALC> tmpBuffer1,tmpBuffer2,tmpBuffer3,tmpBuffer4;
    const
    uint32_t lx;
    uint32_t ly;
    uint32_t lz;
    uint32_t blockNum;
    uint32_t ubSize;
    uint32_t blockLength;
    uint32_t dim;
    uint32_t dimNum;    //目标维数上的长度
    // uint32_t tileNum;
    // uint32_t tileLength;
};
extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelSArg<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x,indice,values,tiling_data.lx,tiling_data.ly,tiling_data.lz,tiling_data.blockNum,tiling_data.ubSize,tiling.dim,tiling.dimNum);
    op.process();
}