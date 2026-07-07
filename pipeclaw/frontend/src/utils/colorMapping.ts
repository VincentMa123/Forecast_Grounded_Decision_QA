/**
 * 管道划分颜色映射
 * 根据管道划分字段为不同管道分配不同颜色
 * 使用38种饱和色，确保每个管道都有独特的颜色
 */
export const PIPELINE_COLORS: Record<string, string> = {
  // 西气东输系列 - 红色系
  '西气东输一线': '#E74C3C',    // 深红色
  '西气东输二线': '#C0392B',    // 石榴红
  '西气东输三线': '#FF6B6B',    // 亮红色

  // 陕京系列 - 橙色系
  '陕京一线': '#F39C12',        // 橙黄色
  '陕京二线': '#E67E22',        // 胡萝卜橙
  '陕京三线': '#D35400',        // 南瓜橙
  '陕京四线': '#FF8C42',        // 亮橙色

  // 中俄/中缅/中贵/中原系列 - 蓝色系
  '中俄东线': '#3498DB',        // 天蓝色
  '中缅线': '#2980B9',          // 深蓝色
  '中贵线': '#5DADE2',          // 亮蓝色
  '中原管道': '#1ABC9C',        // 青绿色

  // 川系列 - 绿色系
  '川气东送': '#2ECC71',        // 翠绿色
  '川东北-川西管道': '#27AE60', // 深绿色

  // 省级管网 - 紫色系
  '广东省管网': '#9B59B6',      // 紫色
  '浙江省管网': '#8E44AD',      // 深紫色
  '山东管道': '#AF7AC5',        // 亮紫色
  '海南管网': '#BB8FCE',        // 淡紫色
  '广西管道': '#7D3C98',        // 暗紫色
  '天津管道': '#D2B4DE',        // 粉紫色
  '重庆管道': '#6C3483',        // 暗紫红

  // 海西系列 - 青色系
  '海西管网': '#1ABC9C',        // 青绿色
  '港清线': '#16A085',          // 墨绿色
  '青宁线': '#48C9B0',          // 浅青色

  // 东北系列 - 粉色系
  '大沈线': '#EC7063',          // 粉红色
  '哈沈线': '#E91E63',          // 玫红色
  '秦沈线': '#F06292',          // 亮粉色

  // 山东系列 - 黄色系
  '泰青威': '#F1C40F',          // 黄色

  // 其他重要管道 - 混合饱和色
  '涩兰长': '#FF5722',          // 深橙色
  '冀宁线': '#00BCD4',          // 青色
  '鄂安沧': '#FFEB3B',          // 亮黄色
  '榆济线': '#4CAF50',          // 绿色
  '忠武线': '#FF9800',          // 橙色
  '苏皖管道': '#03A9F4',        // 亮蓝色
  '长长吉': '#CDDC39',          // 黄绿色

  // 煤制气系列 - 棕色系
  '蒙西煤制气管道': '#795548',  // 棕色
  '新疆煤制气外输管道': '#A1887F', // 浅棕色

  // LNG - 特殊色
  '深圳LNG外输管道': '#00E5FF',// 青蓝色
};

/**
 * 获取管道颜色
 * @param 管道划分 管道划分名称
 * @returns 颜色字符串
 */
export function getPipelineColor(管道划分: string): string {
  return PIPELINE_COLORS[管道划分] || '#999999'; // 默认灰色
}

/**
 * 根据流量计算线条宽度
 * @param 流量 管道流量值
 * @returns 线条宽度（像素）
 */
export function getLineWidth(流量: number): number {
  // 流量范围: 0-5000
  // 线宽范围: 2-8 (最小2像素)
  const minFlow = 0;
  const maxFlow = 5000;
  const minWidth = 2;
  const maxWidth = 8;

  const normalized = Math.min(Math.max(流量, minFlow), maxFlow);
  return minWidth + (normalized / maxFlow) * (maxWidth - minWidth);
}

/**
 * 根据流量计算节点半径
 * @param 流量 节点流量值（计算流量）
 * @returns 圆形半径（像素）
 */
export function getNodeRadius(流量: number): number {
  // 使用绝对值,范围: 0-5000
  // 半径范围: 5-13 (最小5像素)
  const absFlow = Math.abs(流量);
  const minFlow = 0;
  const maxFlow = 5000;
  const minRadius = 5;
  const maxRadius = 13;

  const normalized = Math.min(Math.max(absFlow, minFlow), maxFlow);
  return minRadius + (normalized / maxFlow) * (maxRadius - minRadius);
}

/**
 * 获取所有唯一的管道划分值
 * 用于动态生成颜色映射
 * @param records 管道流量记录数组
 * @returns 唯一管道划分值数组
 */
export function getUniquePipelineDivisions(records: Array<{ 管道划分: string }>): string[] {
  const divisions = new Set<string>();
  records.forEach(record => {
    if (record.管道划分) {
      divisions.add(record.管道划分);
    }
  });
  return Array.from(divisions).sort();
}

/**
 * 为管道划分生成颜色映射表达式（用于MapLibre）
 * @param divisions 管道划分数组
 * @returns MapLibre match表达式
 */
export function generateColorExpression(divisions: string[]): any[] {
  const expression: any[] = ['match', ['get', '管道划分']];

  divisions.forEach(division => {
    const color = getPipelineColor(division);
    expression.push(division, color);
  });

  expression.push('#999999'); // 默认颜色
  return expression;
}
