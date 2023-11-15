let wordBarCharts,
  sentimentBarChart = [];
$(function () {
  const link = location.href;
  d3.json(`${link}word-count/`).then((jsonData) => {
    const countData = getKeywordCount(jsonData);
    wordBarCharts = countData.map((countObj) => {
      return Object.keys(countObj).map((key) => {
        const obj = Object.values(countObj[key])[0];
        const keyword = Object.keys(countObj[key])[0];
        const bc = new Barchart(`#word-count-${key} svg`, keyword.replace(/\s/g, ''));
        bc.initialize();
        updateBarChart(bc, obj);
        return bc;
      })[0];
    });
  });
  const infer = $('.infer');
  const keywords = $.map(infer, (d) => $(d).data('keyword'));
  console.log('keywords', keywords);
  d3.json(`${link}infer/?keywords=${keywords.join(',')}`).then(
    (jsonData) => {
      console.log(jsonData);
      const sentiments = jsonData.sentiments;
      Object.keys(sentiments).forEach((key) => {
        const keyword = key.replace(/\s/g, '');
        console.log('status', sentiments[key].status);
        if (sentiments[key].status === 'error') {
          $(`#infer-${keyword}`).html('감성분석에 에러가 발생했습니다.');
        } else if (sentiments[key].status === 'run') {
          $(`#infer-${keyword}`).html('감성분석 중입니다.');
        } else {
          $(`#infer-${keyword}`).find('.spinner-border').remove();
          const keywordTriplets = jsonData.triplets[key];
          $(`#desc-${keyword}`).text(`총 ${keywordTriplets.length}개의 감성이 추출되었습니다.`);
          renderTriplet(keywordTriplets, keyword);
        }
      });
    },
    (reason) => {
      console.log(reason);
      const spinner = $('.spinner-border');
      spinner.each((d) => {
        $(spinner[d]).text('데이터를 불러오지 못했습니다.');
        $(spinner[d]).removeClass('spinner-border');
      });
    }
  );
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
  const sentimentCount = countSentiments(triplets, keyword);
  console.log(sentimentCount);
  const bc = new Barchart(`#sentiment-${keyword} svg`, keyword);
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
        ><div>${raw}</div>
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

function highlight(xVars, keyword) {
  const x = xVars[0];
  const polarities = ['positive', 'negative', 'neutral'];
  const polMap = { positive: 'pos', negative: 'neg', neutral: 'neu' };

  if (polarities.includes(x)) {
    const target = polMap[x];
    const container = $(`#accordion-${keyword} .accordion-item`);
    container.each(function () {
      const ele = $(this);
      // 키워드 필터 제거
      const btn = ele.find('.accordion-button');
      const originalText = btn.html();
      btn.find('.text-danger').each(function () {
        const beforeText = $(this).text();
        btn.html(originalText.replace(new RegExp(`<span class="text-danger">${beforeText}</span>`, 'g'), beforeText));
      });
      ele.css('display', 'block');

      // 감성 필터 적용
      if (!ele.text().includes(target)) {
        if (!ele.hasClass('hide')) {
          ele.addClass('hide');
        }
      } else {
        if (ele.hasClass('hide')) {
          ele.removeClass('hide');
        }
      }
    });
    const filter = $(`#filter-${keyword}`);
    if (filter.text() !== x) {
      filter.text(x);
      if (x == 'positive') {
        filter.attr('class', 'ms-2 badge bg-primary');
      } else if (x == 'negative') {
        filter.attr('class', 'ms-2 badge bg-danger');
      } else if (x == 'neutral') {
        filter.attr('class', 'ms-2 badge bg-secondary');
      }
    } else {
      filter.text('');
    }
  } else {
    const container = $(`#accordion-${keyword} .accordion-item`);
    const filter = $(`#filter-${keyword}`);
    if (filter.text() !== x) {
      // 다르면 다끄고 재설정
      filter.text(x);
      filter.attr('class', 'ms-2 badge bg-info');
      container.each(function () {
        const ele = $(this);
        const btn = ele.find('.accordion-button');
        const originalText = btn.html();
        btn.find('.text-danger').each(function () {
          const beforeText = $(this).text();
          btn.html(originalText.replace(new RegExp(`<span class="text-danger">${beforeText}</span>`, 'g'), beforeText));
        });
        ele.css('display', 'block'); // 다 보이게
      });
      container.each(function () {
        const ele = $(this);
        const btn = ele.find('.accordion-button');
        if (btn.text().includes(x)) {
          // 단어 포함된 문장
          ele.css('display', 'block');
          const originalText = btn.html();
          btn.html(originalText.replace(new RegExp(x, 'g'), `<span class="text-danger">${x}</span>`));
        } else {
          ele.css('display', 'none');
        }
      });
    } else {
      // 같으면 토글
      filter.text('');
      container.each(function () {
        const ele = $(this);
        const btn = ele.find('.accordion-button');
        if (btn.text().includes(x)) {
          // 단어 포함된 문장
          ele.css('display', 'block');
          const originalText = btn.html();
          if (originalText.includes(`<span class="text-danger">${x}</span>`)) {
            // 만약 같은 단어가 강조된 상태면
            btn.html(originalText.replace(new RegExp(`<span class="text-danger">${x}</span>`, 'g'), x));
          } else {
            btn.html(originalText.replace(new RegExp(x, 'g'), `<span class="text-danger">${x}</span>`));
          }
        } else {
          if (ele.css('display') === 'none') {
            ele.css('display', 'block');
          } else {
            ele.css('display', 'none');
          }
        }
      });
    }
  }
}
