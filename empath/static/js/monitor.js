$(function () {
  const badge = (text) => {
    const badge = $('<span class="badge bg-primary me-1"></span>').text(text);
    const closeButton = $(
      '<button type="button" class="btn-sm btn-close btn-close-white ms-2" aria-label="Close" style="width: 9px;height: 9px;"></button>'
    );

    closeButton.click(() => {
      badge.remove();
      const badgeText = badge.text().trim();
      keywords = keywords.filter((text) => text !== badgeText);
      let storageKeywords = localStorage.getItem('keywords');
      const newKeywords = storageKeywords
        .split(',')
        .filter((text) => text !== badgeText)
        .join(',');
      localStorage.setItem('keywords', newKeywords);
    });

    badge.append(closeButton);

    return badge;
  };

  // 모니터링 정렬 드롭다운 문구 변경
  const sortText = $(`.dropdown-item[href="/crawler/task/${location.search}"]`).text();
  $('#sort').text(sortText);

  const fromInput = $('#from');
  const endInput = $('#end');

  // Task 생성 date input 날짜 제한
  const today = new Date().toISOString().split('T')[0];
  fromInput.prop('max', today);
  endInput.prop('max', today);
  let keywords = [];
  let storageKeywords = localStorage.getItem('keywords');
  if (storageKeywords) keywords = storageKeywords.split(',');
  keywords.forEach((keyword) => {
    $('#keywordContainer').append(badge(keyword));
  });

  $('#addKeyword').click(function (e) {
    e.preventDefault();
    let keyword = $('#keywords').val();
    if (keywords.includes(keyword)) {
      alert('이미 존재하는 키워드입니다.');
      return;
    } else if (keyword === '') {
      alert('키워드가 입력되지 않았습니다.');
      return;
    } else if (keyword.indexOf(',') !== -1) {
      alert('키워드에 콤마(,)가 포함되면 안됩니다.');
      return;
    } else {
      keywords.push(keyword);
      $('#keywordContainer').append(badge(keyword));
      localStorage.setItem('keywords', keywords.join(','));
      $('#keywords').val(''); // 입력 필드 초기화
      console.log($('#keywords').val());
    }
  });

  $('#platform').change(function () {
    if (this.value === 'news') {
      fromInput.prop('disabled', false);
      endInput.prop('disabled', false);
    } else {
      fromInput.prop('disabled', true);
      endInput.prop('disabled', true);
    }
  });
  endInput.change(function () {
    fromInput.prop('max', endInput.val());
    if (fromInput.val() > endInput.val()) fromInput.val(endInput.val());
  });

  $('#taskForm').submit(function (e) {
    e.preventDefault();

    keywords = keywords.map((kw) => {
      return kw.trim();
    });
    if (keywords.length === 0) {
      alert('키워드를 입력하지 않았습니다.');
      return;
    }
    if ($('#platform').val() === '') {
      alert('플랫폼을 선택하지 않았습니다.');
      return;
    }

    console.log('keywords', keywords);
    console.log('platform', $('#platform').val());

    if (confirm('Task를 생성하시겠습니까?')) {
      const data = {
        platform: $('#platform').val(),
        keywords: keywords.join(','),
        name: $('#name').val(),
        from: $('#from').val(),
        end: $('#end').val(),
        limit: $('#limit').val()
      };

      $.ajax({
        url: '/crawler/task/create/',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(data),
        success: function (response) {
          console.log('Success:', response);
          if (response.status === 'success') {
            localStorage.removeItem('keywords');
          } else {
            alert(response.message);
          }
        },
        error: function (error) {
          console.error('Error:', error);
          alert(error);
        }
      });
    }
  });
});

const changeName = (taskId) => {
  const target = `task-${taskId}`;
  const beforeName = $(`#${target}`).text();
  $(`#${target}`).html(
    `<input type="text" id="new-name-${target}" class="form-control-sm" value="${beforeName}" data-before="${beforeName}"></input>`
  );
  $('.card-footer .btn-light').prop('disabled', true);
  $(`#btn-${target}`).removeAttr('onclick');
  $(`#btn-${target}`).attr('onclick', `postTaskName(${taskId})`);
  $(`#btn-${target}`).prop('disabled', false);
  $(`#btn-${target}`).text('저장');
};

const postTaskName = (taskId) => {
  const target = `task-${taskId}`;
  console.log('postTaskName');
  const newName = $(`#new-name-${target}`).val().trim();
  const beforeName = $(`#new-name-${target}`).data('before').trim();
  if (newName === beforeName) {
    return;
  }
  console.log('newName', newName);
  console.log('beforeName', beforeName);
  const data = { id: taskId, name: newName };
  $.ajax({
    url: '/crawler/task/name-change/',
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify(data),
    success: function (response) {
      if (response.status === 'success') {
        $(`#${target}`).text(newName);
        alert('이름 변경 성공');
      } else {
        $(`#${target}`).text(beforeName);
        alert('이름 변경 실패');
      }
      resetTaskName(taskId);
    },
    error: function (error) {
      alert(error);
      $(`#${target}`).text(beforeName);
      resetTaskName(taskId);
    }
  });
};

function resetTaskName(taskId) {
  const target = `task-${taskId}`;
  $('.card-footer .btn-light').prop('disabled', false);
  $(`#btn-${target}`).removeAttr('onclick');
  $(`#btn-${target}`).attr('onclick', `changeName('${taskId}')`);
  $(`#btn-${target}`).text('이름 변경');
}
