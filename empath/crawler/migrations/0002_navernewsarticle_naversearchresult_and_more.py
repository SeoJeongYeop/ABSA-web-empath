# Generated by Django 4.2.4 on 2023-09-08 13:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('crawler', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='NaverNewsArticle',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('oid', models.CharField(max_length=4)),
                ('aid', models.CharField(max_length=20)),
                ('title', models.CharField(max_length=100)),
                ('content', models.TextField()),
                ('published_at', models.DateTimeField()),
                ('press_name', models.CharField(max_length=20)),
                ('crawled_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='NaverSearchResult',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=100)),
                ('keyword', models.CharField(max_length=100)),
                ('press_name', models.CharField(max_length=20)),
                ('link', models.CharField(max_length=200)),
                ('summary', models.CharField(max_length=200)),
                ('crawled_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.RenameModel(
            old_name='SearchResult',
            new_name='YoutubeSearchResult',
        ),
    ]
