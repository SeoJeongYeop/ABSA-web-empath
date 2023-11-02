let wordBarCharts,
  sentimentBarChart = [];
$(function () {
  const link = location.href;
  d3.json(`${link}word-count/`).then((jsonData) => {
    const countData = getKeywordCount(jsonData);

    wordBarCharts = countData.map((countObj) => {
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
          $(`#infer-${keyword} .spinner-border`).remove();
          renderTriplet(jsonData.triplets[keyword], keyword);
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

function renderTriplet(triplets, keyword) {
  console.log('renderTriplet', triplets);
  const sentimentCount = countSentiments(triplets, keyword);
  console.log(sentimentCount);
  const bc = new Barchart(`#sentiment-${keyword} svg`);
  bc.initialize();
  updateBarChart(bc, sentimentCount);
  sentimentBarChart.push(bc);
}

function countSentiments(data, keyword) {
  const keywordData = data;
  const sentimentCounts = {
    positive: 0,
    negative: 0,
    neutral: 0
  };
  let body = '';

  keywordData.forEach((item, i) => {
    const polarities = item.polarities;
    const aspects = item.aspects;
    const opinions = item.opinions;
    const raw = item.raw_sentence;

    for (let i = 0; i < polarities.length; i++) {
      const pol = polarities[i];
      const asp = aspects[i];
      const opn = opinions[i];
      let polClass;
      if (pol === '<pos>') {
        sentimentCounts.positive++;
        polClass = 'primary';
      } else if (pol === '<neg>') {
        sentimentCounts.negative++;
        polClass = 'danger';
      } else if (pol === '<neu>') {
        sentimentCounts.neutral++;
        polClass = 'secondary';
      } else {
        continue;
      }
      body += `<div class="accordion-body">
          <span class="badge bg-pastel-${polClass} text-${polClass}">
          ${pol.slice(1, 4)}
          </span>
          <div>${asp} <i class="bi bi-arrow-right-short"></i>${opn}</div>
        </div>`;
    }
    $(`#accordion-${keyword}`).append(
      `<div class="accordion-item">
      <h2 class="accordion-header">
        <button
          class="accordion-button gap-2 collapsed"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#col-${keyword}-${i}"
          aria-expanded="false"
          aria-controls="col-${keyword}-${i}"
        >
          ${raw}
        </button>
      </h2>
      <div id="col-${keyword}-${i}" class="accordion-collapse collapse bg-gray-200" data-bs-parent="#accordion-${keyword}">
        ${body}
      </div>
    </div>`
    );
    body = '';
  });

  return sentimentCounts;
}
