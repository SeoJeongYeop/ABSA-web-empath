let BarCharts;
$(function () {
  const link = location.href;
  d3.json(`${link}word-count/`).then((jsonData) => {
    const countData = getKeywordCount(jsonData);

    BarCharts = countData.map((countObj) => {
      return Object.keys(countObj).map((key) => {
        const obj = Object.values(countObj[key])[0];
        const bc = new Barchart(`#word-count-${key} svg`);
        bc.initialize();
        updateBarChart(bc, obj);
        return bc;
      })[0];
    });
  });
  const infer = $('.infer');
  console.log('infer', infer);
  const keywords = $.map(infer, (d) => $(d).data('keyword'));

  // const keywords = infer.each((d) => d.data('keyword'));
  console.log('keywords', keywords.join(','));
  d3.json(`${link}infer/?keywords=${keywords.join(',')}`).then(
    (jsonData) => {
      console.log(jsonData);
      const sentiments = jsonData.sentiments;
      Object.keys(sentiments).forEach((keyword) => {
        console.log('status', sentiments[keyword].status);
        if (sentiments[keyword].status === 'error') {
          $(`#infer-${keyword}`).html('감성분석에 에러가 발생했습니다.');
        } else if (sentiments[keyword].status === 'run') {
          $(`#infer-${keyword}`).html('감성분석 중입니다.');
        } else {
          renderTriplet(jsonData.triplets);
        }
      });
    },
    (reason) => {
      console.log(reason);
      const spinner = $('.spinner-border');
      console.log('spinner', spinner);
      spinner.each((d) => {
        console.log('d', spinner[d]);
        $(spinner[d]).text('데이터를 불러오지 못했습니다.');
        $(spinner[d]).removeClass('spinner-border');
      });
    }
  ).f;
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

function renderTriplet(triplets) {
  console.log('renderTriplet', triplets);
}
