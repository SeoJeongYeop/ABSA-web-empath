let BarCharts;
$(function () {
  const link = location.href;
  d3.json(`${link}word-count/`).then((jsonData) => {
    const countData = getKeywordCount(jsonData);

    BarCharts = countData.map((countObj) => {
      return Object.keys(countObj).map((key) => {
        const obj = Object.values(countObj[key])[0];
        BarChart = new Barchart(`#word-count-${key} svg`);
        BarChart.initialize();
        updateBarChart(BarChart, obj);
        return BarChart;
      })[0];
    });
  });
});

function getKeywordCount(data) {
  const ret = Object.values(data).map((obj) => {
    const keywordCount = {};
    keywordCount[obj.id] = {};
    keywordCount[obj.id][obj.keyword] = {};
    obj.token_count.forEach((d) => {
      keywordCount[obj.id][obj.keyword][d[0]] = d[1];
    });
    return keywordCount;
  });

  return ret;
}
/**
 * bar 차트 업데이트
 */
function updateBarChart(chart, obj) {
  if (obj === null) chart.delete();
  else chart.update(obj, Object.keys(obj));
}
