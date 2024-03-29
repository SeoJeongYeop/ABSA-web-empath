# Generated by Django 4.2.4 on 2023-09-05 06:27

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SearchResult',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('keyword', models.CharField(max_length=100)),
                ('video_id', models.CharField(max_length=20)),
                ('crawled_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='YoutubeChannel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('channel_id', models.CharField(db_index=True, max_length=32, unique=True)),
                ('title', models.CharField(max_length=100)),
                ('description', models.TextField()),
                ('published_at', models.DateTimeField()),
                ('subscriber_count', models.IntegerField()),
                ('video_count', models.IntegerField()),
                ('crawled_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='YoutubeVideo',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('video_id', models.CharField(db_index=True, max_length=20, unique=True)),
                ('title', models.CharField(max_length=200)),
                ('published_at', models.DateTimeField()),
                ('description', models.TextField()),
                ('channel_id', models.CharField(max_length=32)),
                ('channel_title', models.CharField(max_length=100)),
                ('tags', models.TextField()),
                ('category_id', models.IntegerField()),
                ('comment_count', models.IntegerField()),
                ('crawled_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='YoutubeComment',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('comment', models.TextField()),
                ('crawled_at', models.DateTimeField(auto_now_add=True)),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='crawler.youtubevideo')),
            ],
        ),
    ]
