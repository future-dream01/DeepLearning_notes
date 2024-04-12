void FOVE156()
{

// 将.root文件中的值分配给对应的leaf
TLeaf *rsector1=Coincidences->GetLeaf("rsectorID1");       
TLeaf *crystal1=Coincidences->GetLeaf("crystalID1");
TLeaf *module1=Coincidences->GetLeaf("moduleID1");
TLeaf *rsector2=Coincidences->GetLeaf("rsectorID2");
TLeaf *crystal2=Coincidences->GetLeaf("crystalID2");
TLeaf *module2=Coincidences->GetLeaf("moduleID2");


FILE *fp1=fopen("xlmx_3col_800_1s_1.doc","w");
FILE *fp2=fopen("xlmx_3col_800_1s_2.doc","w");

const int DUISHU=156;                       // 定义探测单元中总的resector对数
const int QIAN =52;                         // 合并前的层数
const int HOU =25;                          // 合并后的层数

double FIG1[HOU][QIAN][DUISHU][DUISHU]={0};    //创建四维张量，包含156对数据，总共52个切片图，其中每两个合成一个
double FIG2[HOU][QIAN][DUISHU][DUISHU]={0};

getmax=Coincidences->GetEntries();              // 获取所有符合数据的数量
timemax=getmax;                                 //得到最大遍历次数

/////////////////////////////////////////////////////
// 将仿真结果.root文件中的数据写入两个张量FIG1、FIG2中//
////////////////////////////////////////////////////
for(int i=0;i<=timemax;i++)     // 读取符合数据的ID，即符合线两端点的位置，确定符合线
{
Coincidences->GetEntry(i);      // 加载第i个事件的数据
r1=rsector1->GetValue();        // GetValue()方法获取参数
c1=crystal1->GetValue();
m1=module1->GetValue();
r2=rsector2->GetValue();
c2=crystal2->GetValue();
m2=module2->GetValue();

// 根据符合线的端点ID确定符合线端点对应哪个探测环、在环上哪个位置
c1m=(int)c1;                   // 将c1转换成整数
c2m=(int)c2;
c1use=c1m/13;
c2use=c2m/13;
axial1=m1*13+c1use+1;           // 计算轴向位置
axial2=m2*13+c2use+1;           // 计算轴向位置

// 计算符合线中点所处的探测环位置以及中间隔了几个探测环，用于判断是否剔除该符合线
asub=(axial1-axial2)/4;         
amid=(axial1+axial2)/2-1;
asubout=asub+12;

cx1=c1m%13;
cx2=c2m%13;

// 获取投影所对应的符合线在投影切片上的位置
x1=13*r1+1+cx1-7;               // 计算横截面位置
x2=13*r2+1+cx2-7;               // 计算轴向位置

if(x1<1)
  x1=x1+2*DUISHU;
if(x1<1)
  x2=x2+2*DUISHU;

// 根据符合线的投影ID 计算两个变量，为后续计算做铺垫
xsum=(x1+x2)/2;
xsub=abs(x1-x2);
a2=int(x1+x2);
a1=a2%2;

// 确定该投影符合线在投影面所对应正弦图的位置
  if(a1==0)  
  {   
    if(xsum<=DUISHU)
    {
    xt=xsum;
    xr=xsub/2;
    }
    else
    {
    xt=xsum-DUISHU;
    xr=abs(2*DUISHU-xsub)/2;
    }

  FIG1[asubout][amid][xt-1][xr-1]=FIG1[asubout][amid][xt-1][xr-1]+1;
  
  };
  else
  {
    xsum=xsum+1;
    if(xsum<=DUISHU)
    {
    xt=xsum;
    xr=xsub/2+1;
    }
    else
    {
    xt=xsum-DUISHU;
    xr=abs(2*DUISHU-xsub)/2+1;
    }

  FIG2[asubout][amid][xt-1][xr-1]=FIG2[asubout][amid][xt-1][xr-1]+1;

  } 
};                          // 以上将原始的.root文件成功读取到两个张量FIG1、FIG2中


/////////////////////////////////////////////////////
//////// 将两个张量中的数据写入两个doc文件中 ///////////
/////////////////////////////////////////////////////
for(int ls=0;ls<HOU;ls++)
{
for(int k=0;k<QIAN;k++)
{
for(int i=0;i<DUISHU;i++)   
{
  for(int j=0;j<DUISHU;j++)
  {
  fprintf(fp1,"%f  ",FIG1[ls][k][i][j]);    // 打开fp1，将这个张量写入fp1
  }
  fprintf(fp1,"\n");                        // 换行
};
};
};
fclose(fp1);                                // 关闭文件

for(int ls=0;ls<HOU;ls++)
{
for(int k=0;k<QIAN;k++)
{
for(int i=0;i<DUISHU;i++)   
{
  for(int j=0;j<DUISHU;j++)
  {
  fprintf(fp2,"%f  ",FIG2[ls][k][i][j]);     // 打开fp1，将这个张量写入fp1
  }
  fprintf(fp2,"\n");                         // 换行
};
};
};
fclose(fp2);                                 // 关闭文件

}
