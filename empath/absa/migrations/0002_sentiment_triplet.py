# Generated by Django 4.2.4 on 2023-11-01 14:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('crawler', '0010_navernewsarticle_keyword'),
        ('absa', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Sentiment',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('keyword', models.CharField(max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(max_length=20)),
                ('task', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='crawler.task')),
            ],
        ),
        migrations.CreateModel(
            name='Triplet',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('raw_sentence', models.TextField()),
                ('aspects', models.TextField()),
                ('opinions', models.TextField()),
                ('polarities', models.TextField()),
                ('sentiment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='absa.sentiment')),
                ('source_news', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='crawler.navernewsarticle')),
                ('source_youtube', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='crawler.youtubevideocomment')),
            ],
        ),
    ]