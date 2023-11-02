class Barchart {
  margin = {
    top: 10,
    right: 10,
    bottom: 30,
    left: 30
  };

  constructor(svg, keyword, width = 400, height = 240) {
    this.svg = svg;
    this.width = width;
    this.height = height;
    this.keyword = keyword;
  }

  initialize() {
    this.svg = d3.select(this.svg);
    this.container = this.svg.append('g');
    this.xAxis = this.svg.append('g');
    this.yAxis = this.svg.append('g');
    this.legend = this.svg.append('g');

    this.xScale = d3.scaleBand();
    this.yScale = d3.scaleLinear();
    this.zScale = d3.scaleOrdinal().range(PALETTE);

    this.svg
      .attr('width', this.width + this.margin.left + this.margin.right)
      .attr('height', this.height + this.margin.top + this.margin.bottom);

    this.container.attr('transform', `translate(${this.margin.left}, ${this.margin.top})`);
  }

  update(data, xVars) {
    this.svg.style('display', 'block');

    const categories = [...new Set(xVars)];
    this.xScale.domain(categories).range([0, this.width]).padding(0.3);
    this.yScale.domain([0, d3.max(categories.map((c) => data[c]))]).range([this.height, 0]);

    this.container
      .selectAll('rect')
      .data(categories)
      .join('rect')
      .attr('x', (d) => this.xScale(d))
      .attr('y', (d) => this.yScale(data[d]))
      .attr('width', this.xScale.bandwidth())
      .attr('height', (d) => this.height - this.yScale(data[d]))
      .attr('fill', (d) => this.zScale(d))
      .on('click', (d, i) => {
        highlight(
          categories.filter((idx) => idx === i),
          this.keyword
        );
      });

    this.xAxis
      .attr('transform', `translate(${this.margin.left}, ${this.margin.top + this.height})`)
      .style('font-size', '0.8rem')
      .call(d3.axisBottom(this.xScale));

    this.yAxis
      .attr('transform', `translate(${this.margin.left}, ${this.margin.top})`)
      .style('font-size', '0.9rem')
      .call(d3.axisLeft(this.yScale));
  }

  delete() {
    this.container.selectAll('rect').remove();
    this.svg.style('display', 'none');
  }
}
PALETTE = [
  '#1f77b4',
  '#d62728',
  '#ff7f0e',
  '#2ca02c',
  '#9467bd',
  '#8c564b',
  '#e377c2',
  '#7f7f7f',
  '#bcbd22',
  '#17becf'
];
